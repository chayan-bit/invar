//! Physical twelve-buffer ABI5 ownership. This boundary accepts compiler-owned
//! source/layout only; root performs source/type/provenance/fence admission.
use super::*;
use cudarc::driver::{DevicePtr, HostSlice, SyncOnDrop, result};

/// Page-locked host words in ordinary cacheable memory: the workset's upload
/// and readback staging. cudarc's `alloc_pinned` maps write-combined pages,
/// which the CPU reads uncached (about 0.19 GB/s measured on an RTX A6000
/// host), and every epoch reads the whole workset back twice.
struct Staging {
    ptr: *mut u64,
    len: usize,
    stream: Arc<CudaStream>,
}
// SAFETY: Staging exclusively owns its allocation, as PinnedHostSlice does.
unsafe impl Send for Staging {}
unsafe impl Sync for Staging {}
impl Staging {
    fn zeroed(stream: &Arc<CudaStream>, words: usize) -> Result<Self, DriverError> {
        stream.context().bind_to_thread()?;
        // SAFETY: flags 0 is cacheable page-locked memory; it is zeroed below
        // before any use and freed only by Drop.
        let ptr = unsafe { result::malloc_host(words * 8, 0) }?.cast::<u64>();
        assert!(!ptr.is_null() && ptr.is_aligned());
        // SAFETY: the allocation holds exactly `words` writable u64 slots.
        unsafe { std::ptr::write_bytes(ptr, 0, words) };
        Ok(Self {
            ptr,
            len: words,
            stream: stream.clone(),
        })
    }
    /// Callers read only after the work's event confirmed every transfer.
    fn as_slice(&self) -> &[u64] {
        // SAFETY: `len` initialized words owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// Callers write only while no transfer uses these pages.
    fn as_mut_slice(&mut self) -> &mut [u64] {
        // SAFETY: `len` initialized words exclusively borrowed from self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}
impl HostSlice<u64> for Staging {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [u64], SyncOnDrop<'a>) {
        // Work orders every transfer before its own recorded event.
        (self.as_slice(), SyncOnDrop::Sync(None))
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [u64], SyncOnDrop<'a>) {
        (self.as_mut_slice(), SyncOnDrop::Sync(None))
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        // A transfer may still target the pages: free them only after the
        // stream drained, and leak them if that cannot be confirmed.
        if self.stream.synchronize().is_ok() {
            // SAFETY: allocated by malloc_host in `zeroed` and freed once.
            let _ = unsafe { result::free_host(self.ptr.cast()) };
        }
    }
}

pub struct Work {
    buffers: Vec<CudaSlice<u64>>,
    staging: Vec<Staging>,
    function: CudaFunction,
    resolver: Option<CudaFunction>,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    event: CudaEvent,
    addresses: [u64; 12],
    bytes: [usize; 12],
    submitted: bool,
    geometry: (u32, u32),
}
impl Work {
    pub fn addresses(&self) -> [u64; 12] {
        self.addresses
    }
    pub fn context_identity(&self) -> u64 {
        self.context.cu_ctx() as usize as u64
    }
    pub fn bytes(&self) -> [usize; 12] {
        self.bytes
    }
    /// Launch and all twelve readback transfers precede the same observed fence.
    pub fn submit(&mut self) -> Result<(), Error> {
        // A prior epoch's event can never confirm this new submission.
        self.submitted = false;
        let mut launch = self.stream.launch_builder(&self.function);
        for buffer in &mut self.buffers {
            launch.arg(buffer);
        }
        // SAFETY: root admitted the immutable scalar ABI5 layout, twelve real
        // distinct 8-byte aligned owned allocations, one32-thread group and
        // finite configuration. Work retains buffers/module/context/queue and
        // pinned readback storage through the final event. No external pointers.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (self.geometry.0, 1, 1),
                block_dim: (self.geometry.1, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(driver_error)?;
        if let Some(resolver) = &self.resolver {
            let mut reduce = self.stream.launch_builder(resolver);
            for buffer in &mut self.buffers {
                reduce.arg(buffer);
            }
            // SAFETY: same retained twelve-buffer capsule, ordered after its
            // producers in this stream. Exactly 256 lanes own shared reduction.
            unsafe {
                reduce.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(driver_error)?;
        }
        for (buffer, host) in self.buffers.iter().zip(&mut self.staging) {
            self.stream
                .memcpy_dtoh(buffer, host)
                .map_err(driver_error)?;
        }
        self.event.record(&self.stream).map_err(driver_error)?;
        self.submitted = true;
        Ok(())
    }
    pub fn complete(&self) -> bool {
        self.submitted && self.event.is_complete()
    }
    pub fn read(&self) -> Result<[Vec<u64>; 12], Error> {
        if !self.complete() {
            return Err(Error::Poisoned);
        }
        let mut outputs: [Vec<u64>; 12] = std::array::from_fn(|_| Vec::new());
        for (out, host) in outputs.iter_mut().zip(&self.staging) {
            let values = host.as_slice();
            out.try_reserve_exact(values.len())
                .map_err(|_| Error::OutOfMemory)?;
            out.extend_from_slice(values);
        }
        Ok(outputs)
    }
}
/// The ABI5 kernel of one compiled-ahead image, loaded by the Driver API
/// alone. Its function owns the module, so in-flight work keeps it alive.
#[derive(Clone)]
pub struct Kernel {
    function: CudaFunction,
    context: Arc<CudaContext>,
}
impl CudaDevice {
    /// Load a caller-validated CUBIN with `cuModuleLoadData`. No compiler runs:
    /// an image the device or driver cannot load is the driver's typed
    /// refusal, never a recompilation.
    pub fn load_gnet_image(&mut self, image: &[u8]) -> Result<Kernel, Error> {
        self.admit()?;
        if image.is_empty() || image.len() > MAX_GNET_CUBIN_BYTES {
            return Err(Error::Limit);
        }
        let module = self
            .context
            .load_module(Ptx::from_binary(image.to_vec()))
            .map_err(driver_error)?;
        let function = module
            .load_function("invar_gnet_children_v5")
            .map_err(driver_error)?;
        if function.max_threads_per_block().map_err(driver_error)? < 32 {
            return Err(Error::Limit);
        }
        Ok(Kernel {
            function,
            context: self.context.clone(),
        })
    }
    /// Production route: a loaded compiled-ahead kernel, no compiler involved.
    pub fn prepare_gnet(
        &mut self,
        kernel: &Kernel,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
    ) -> Result<Work, Error> {
        self.prepare_gnet_launch(bytes, input, config, None, |device| {
            if !Arc::ptr_eq(&kernel.context, &device.context) {
                return Err(Error::InvalidHandle);
            }
            Ok((kernel.function.clone(), None))
        })
    }
    /// P2 region route: compiles the generated source at run time, like the
    /// runtime-NVRTC adapter. Compiled-ahead images never take this path.
    fn prepare_gnet_region(
        &mut self,
        source: &str,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
        compiler: &Path,
        groups: u32,
    ) -> Result<Work, Error> {
        self.prepare_gnet_launch(bytes, input, config, Some(groups), |device| {
            if source.len() > invar_device::MAX_GNET_SOURCE_BYTES {
                return Err(Error::Limit);
            }
            let cubin = compile_gnet_cubin_bounded(source, compiler, device.caps)?;
            let module = device
                .context
                .load_module(Ptx::from_binary(cubin.image))
                .map_err(driver_error)?;
            let function = module
                .load_function("invar_gnet_body_map_v1")
                .map_err(driver_error)?;
            if function.max_threads_per_block().map_err(driver_error)? < 256 {
                return Err(Error::Limit);
            }
            let resolve = module
                .load_function("invar_gnet_body_resolve_v1")
                .map_err(driver_error)?;
            if resolve.max_threads_per_block().map_err(driver_error)? < 256 {
                return Err(Error::Limit);
            }
            Ok((function, Some(resolve)))
        })
    }
    fn prepare_gnet_launch(
        &mut self,
        bytes: [usize; 12],
        input: &[u64],
        config: [u64; 6],
        region_groups: Option<u32>,
        load: impl FnOnce(&Self) -> Result<(CudaFunction, Option<CudaFunction>), Error>,
    ) -> Result<Work, Error> {
        self.admit()?;
        if bytes.iter().any(|n| *n == 0 || n % 8 != 0)
            || bytes
                .iter()
                .try_fold(0usize, |a, n| a.checked_add(*n))
                .is_none_or(|n| n > invar_device::MAX_GNET_STORAGE_BYTES)
            || input.len() > bytes[0] / 8
            || bytes[3] != 48
        {
            return Err(Error::Limit);
        }
        let total = bytes
            .iter()
            .try_fold(0usize, |a, n| a.checked_add(*n))
            .ok_or(Error::Limit)?;
        // Device buffers, pinned readback and host result storage coexist.
        if total
            .checked_mul(3)
            .is_none_or(|n| n > invar_device::MAX_GNET_STORAGE_BYTES)
        {
            return Err(Error::Limit);
        }
        let (free, _) = self.context.mem_get_info().map_err(driver_error)?;
        if total > free.saturating_sub((free / 10).max(1)) {
            return Err(Error::Limit);
        }
        let (function, resolver) = load(self)?;
        let event = self.context.new_event(None).map_err(driver_error)?;
        let mut work = Box::new(Work {
            buffers: Vec::new(),
            staging: Vec::new(),
            function,
            resolver,
            stream: self.stream.clone(),
            context: self.context.clone(),
            event,
            addresses: [0; 12],
            bytes,
            submitted: false,
            geometry: (
                region_groups.unwrap_or(1),
                if region_groups.is_some() { 256 } else { 32 },
            ),
        });
        let prepared = (|| -> Result<(), Error> {
            // Primary buffers0..6 are allocated before any private slots7..11.
            for n in bytes {
                let words = n / 8;
                // Work retains each staging page and every asynchronous
                // transfer. Readback writes only this exact extent.
                work.staging
                    .push(Staging::zeroed(&self.stream, words).map_err(driver_error)?);
                work.buffers.push(
                    self.stream
                        .alloc_zeros::<u64>(words)
                        .map_err(driver_error)?,
                );
            }
            // Input capacity was validated and config is exactly six words.
            work.staging[0].as_mut_slice()[..input.len()].copy_from_slice(input);
            work.staging[3].as_mut_slice().copy_from_slice(&config);
            for i in [0, 3] {
                self.stream
                    .memcpy_htod(&work.staging[i], &mut work.buffers[i])
                    .map_err(driver_error)?;
            }
            self.stream.synchronize().map_err(driver_error)?;
            for (i, buffer) in work.buffers.iter().enumerate() {
                let (pointer, _guard) = buffer.device_ptr(&self.stream);
                work.addresses[i] = pointer;
            }
            Ok(())
        })();
        if let Err(error) = prepared {
            self.hold(work);
            return Err(error);
        }
        Ok(*work)
    }
    /// Synchronous P2 capsule, including uploads and readback under one real
    /// stream fence. Any uncertain operation is retained in the quarantine.
    pub fn run_gnet_region(
        &mut self,
        source: &str,
        image: &[Vec<u64>; 12],
        groups: usize,
        compiler: &Path,
    ) -> Result<[Vec<u64>; 12], Error> {
        if groups == 0 || groups > 65_535 {
            return Err(Error::Limit);
        }
        let config: [u64; 6] = image[3].as_slice().try_into().map_err(|_| Error::Limit)?;
        let bytes = std::array::from_fn(|i| image[i].len().saturating_mul(8));
        let mut work = Box::new(self.prepare_gnet_region(
            source,
            bytes,
            &image[0],
            config,
            compiler,
            groups as u32,
        )?);
        let result = (|| {
            for (i, words) in image.iter().enumerate() {
                // Idle staging of exactly this word extent; the write finishes
                // before its asynchronous upload starts.
                work.staging[i].as_mut_slice().copy_from_slice(words);
                self.stream
                    .memcpy_htod(&work.staging[i], &mut work.buffers[i])
                    .map_err(driver_error)?;
            }
            work.submit()?;
            let start = std::time::Instant::now();
            while !work.complete() {
                if start.elapsed() > std::time::Duration::from_secs(120) {
                    return Err(Error::Timeout);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            work.read()
        })();
        if result.is_err() {
            self.hold(work);
        }
        result
    }
}
