//! Retained, confined native frame. Driver images never come from the user.
use super::*;
use invar_foreign::device::Driver;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

pub(super) struct Native {
    driver: Driver,
    // Held only for its Drop; declared after `driver`, so the library unloads
    // before its directory goes.
    _directory: Scratch,
}

/// The frame's private build directory, removed on every exit: a failed write,
/// compile, digest, load or first image write as well as a normal drop.
struct Scratch(PathBuf);
impl Scratch {
    fn create(path: PathBuf) -> Result<Self, Error> {
        #[allow(unused_mut)] // mode() is Unix-only
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(&path)
            .map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self(path))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
impl Native {
    pub(super) fn compile(
        artifact: &compiled::Artifact,
        image: &[Vec<u64>; 12],
        capsule: bool,
    ) -> Result<Self, Error> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = NEXT
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| x.checked_add(1))
            .map_err(|_| Error::Limit)?;
        let directory = Scratch::create(
            std::env::temp_dir().join(format!("invar-p2-{}-{nonce}", std::process::id())),
        )?;
        let digest: [u8; 32] = Sha256::digest(artifact.source().as_bytes()).into();
        let symbol = format!(
            "invar_p2_{}",
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        );
        let slot = if capsule { "GnetTask" } else { "GnetSpecSlot" };
        let capsule_literal = usize::from(capsule);
        let auxiliary = if capsule { "gnet_w" } else { "gnet_u" };
        let code = format!(
            r#"
#include <array>
#include <vector>
#include <cstring>
#include <new>
#include <memory>
#define __device__
#define __global__
#define __shared__
#define __syncthreads()
struct Dim {{unsigned x=1,y=1,z=1;}};
Dim gridDim,blockDim{{32,1,1}},threadIdx{{0,0,0}},blockIdx{{0,0,0}};
{source}
struct P2Work {{
 std::array<std::vector<gnet_u>,12> b;std::array<size_t,12> sizes{{}};
 std::vector<GnetTask> tasks,seeds;bool submitted=false;
 void* data(size_t i){{if(i==6)return tasks.data();if(i==7&&{capsule_literal})return seeds.data();return b[i].data();}}
}};
extern "C" __attribute__((visibility("default"))) unsigned {symbol}(unsigned cmd,void** handle,const gnet_u* in,size_t n,gnet_u* out,size_t m) noexcept {{
 if(!handle)return 2;
 try {{
 if(cmd==0){{if(*handle||n<18||m!=12)return 2;auto owner=std::make_unique<P2Work>();auto w=owner.get();
  if(in[6]*8%sizeof(GnetTask)||({capsule_literal}&&in[7]*8!=sizeof(GnetTask))){{return 2;}}
  for(size_t i=0;i<12;++i){{w->sizes[i]=in[i];if(i==6)w->tasks.resize(in[i]*8/sizeof(GnetTask));else if(i==7&&{capsule_literal})w->seeds.resize(1);else w->b[i].resize(in[i],0);out[i]=(gnet_u)w->data(i);}}
  if(n-18>w->b[0].size()||w->b[3].size()!=6){{return 2;}}
  std::memcpy(w->data(0),in+18,(n-18)*8);std::memcpy(w->data(3),in+12,48);*handle=owner.release();return 0;}}
 auto w=static_cast<P2Work*>(*handle);if(!w)return 2;
 if(cmd==1){{for(unsigned lane=0;lane<32;++lane){{threadIdx.x=lane;
  invar_gnet_children_v5((gnet_w*)w->data(0),(gnet_w*)w->data(1),(gnet_u*)w->data(2),(const gnet_u*)w->data(3),(gnet_w*)w->data(4),(gnet_u*)w->data(5),(GnetTask*)w->data(6),({slot}*)w->data(7),(gnet_w*)w->data(8),({auxiliary}*)w->data(9),({auxiliary}*)w->data(10),(gnet_u*)w->data(11));
 }}
 if({capsule_literal}){{auto c=(gnet_u*)w->data(11);auto cfg=(gnet_u*)w->data(3);gnet_u best=cfg[0];for(gnet_u lane=0;lane<cfg[0];++lane)if(c[lane*14+6]!=1&&lane<best)best=lane;c[9]=best;}}
 w->submitted=true;return 0;}}
 if(cmd==2){{if(m!=1)return 2;out[0]=w->submitted;return 0;}}
 size_t total=0;for(auto size:w->sizes)total+=size;
 if(cmd==3){{if(!w->submitted||m!=total)return 2;size_t at=0;for(size_t i=0;i<12;++i){{std::memcpy(out+at,w->data(i),w->sizes[i]*8);at+=w->sizes[i];}}return 0;}}
 if(cmd==5){{if(n!=total)return 2;size_t at=0;for(size_t i=0;i<12;++i){{std::memcpy(w->data(i),in+at,w->sizes[i]*8);at+=w->sizes[i];}}w->submitted=false;return 0;}}
 if(cmd==4){{delete w;*handle=nullptr;return 0;}}return 2;
 }}catch(const std::bad_alloc&){{return 1;}}catch(...){{return 4;}}
}}
"#,
            source = artifact.source()
        );
        let source = directory.0.join("frame.cpp");
        let library = directory
            .0
            .join(format!("frame.{}", std::env::consts::DLL_EXTENSION));
        fs::write(&source, code).map_err(|e| Error::Backend(e.to_string()))?;
        let mut command = Command::new(std::env::var_os("CXX").unwrap_or_else(|| "c++".into()));
        command
            .args([
                "-std=c++20",
                "-O1",
                "-shared",
                "-fPIC",
                "-fvisibility=hidden",
                "-fno-fast-math",
                "-ffp-contract=off",
            ])
            .arg(&source)
            .arg("-o")
            .arg(&library);
        let (ok, _, err) = super::super::super::compiler_child::bounded(
            &mut command,
            &directory.0.join("compiler"),
            Duration::from_secs(90),
        )
        .map_err(Error::Backend)?;
        if !ok {
            return Err(Error::Backend(format!(
                "native P2 compile: {}",
                String::from_utf8_lossy(&err)
            )));
        }
        let hash = invar_foreign::numeric::file_digest(&library)
            .map_err(|e| Error::Backend(format!("native digest:{e:?}")))?;
        let bytes = std::array::from_fn(|i| image[i].len() * 8);
        let config: [u64; 6] = image[3]
            .as_slice()
            .try_into()
            .map_err(|_| Error::Malformed)?;
        let mut driver = Driver::load(&library, hash, &symbol, bytes, &image[0], config)
            .map_err(|e| Error::Backend(format!("native load:{e:?}")))?;
        driver
            .replace(image)
            .map_err(|e| Error::Backend(format!("native image:{e:?}")))?;
        Ok(Self {
            driver,
            _directory: directory,
        })
    }
    pub(super) fn run(&mut self, image: &[Vec<u64>; 12]) -> Result<[Vec<u64>; 12], Error> {
        self.driver
            .replace(image)
            .map_err(|e| Error::Backend(format!("native write:{e:?}")))?;
        self.driver
            .submit()
            .map_err(|e| Error::Backend(format!("native submit:{e:?}")))?;
        if !self
            .driver
            .fence()
            .map_err(|e| Error::Backend(format!("native fence:{e:?}")))?
        {
            return Err(Error::Malformed);
        }
        self.driver
            .read()
            .map_err(|e| Error::Backend(format!("native read:{e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Review finding: a compile step that fails after the directory exists
    /// must not leave the directory or its generated files behind.
    #[test]
    fn a_failed_step_removes_the_scratch_directory() {
        let path =
            std::env::temp_dir().join(format!("invar-p2-scratch-test-{}", std::process::id()));
        let failing = || -> Result<(), Error> {
            let scratch = Scratch::create(path.clone())?;
            fs::write(scratch.0.join("frame.cpp"), "int x;")
                .map_err(|e| Error::Backend(e.to_string()))?;
            assert!(scratch.0.join("frame.cpp").exists());
            Err(Error::Backend("native P2 compile: failed".into()))
        };
        assert!(failing().is_err());
        assert!(!path.exists(), "{} leaked", path.display());
    }
}
