#![forbid(unsafe_code)]

use invar::{
    check, eval, front,
    general_net::{self, Net, compiled, oracle},
};
#[test]
fn issue_gnet_06_current_m4_scalar_array_and_record_f32_fail_closed() {
    for source in [
        "fn unused(x:f32)->f32{return x+1.0;} fn main()->i32{return 7;}",
        "struct Numeric {x:f32} fn main()->i32 {let p=Numeric{x:1.0};if p.x==1.0{return 7;}return 0;}",
        "fn main()->i32 {let xs:Array<f32> = [1.0,2.0];if xs[0u32]==1.0{return 7;}return 0;}",
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        assert_eq!(
            compiled::emit_aggregates(
                &net,
                &proof,
                "main",
                compiled::Target::Cpu,
                compiled::Limits::default(),
                compiled::OwnedLimits::default()
            )
            .unwrap_err(),
            compiled::Error::UnsupportedType
        );
    }
}
#[test]
fn issue_gnet_06_current_source_reference_matches_net_every_aggregate_fuel_prefix() {
    for source in [
        "struct Packet {items:Array<i32>,code:i32} fn make()->Packet {let p=Packet{items:[1,2],code:7};return p;} fn consume(p:take Packet)->i32{return p.code;} fn main()->i32 {let p=make();return consume(move p);}",
        "struct Packet {items:Array<i32>,code:i32} fn main()->i32 {if true {let done=Packet{items:[1,2],code:0};}let p=Packet{items:[3],code:7};return p.code/0;}",
        "fn leaf(ok:Bool)->Result<i32,u32> {if ok{return Result::Ok(7);}return Result::Err(9u32);} fn wrap(ok:Bool)->Result<i32,u32> {let scratch:Array<i32> = [1,2];let n:i32=leaf(ok)?;return Result::Ok(n);} fn main()->i32 {let r=wrap(false);return match move r {Result::Ok(n)=>n,Result::Err(e)=>9};}",
    ] {
        let checked = check::check(front::parse(source.as_bytes()).unwrap(), true).unwrap();
        let net = Net::lower(&checked, general_net::Limits::default()).unwrap();
        let proof = net.verify().unwrap();
        let complete = eval::observe(&checked, 10000);
        for fuel in 1..=complete.steps_used + 1 {
            let reference = eval::observe(&checked, fuel);
            let actual = oracle::observe(
                &net,
                &proof,
                oracle::Limits {
                    fuel,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(actual.failure, None);
            assert_eq!(
                actual.semantic.result, reference.result,
                "fuel{fuel}: {source}"
            );
            assert_eq!(actual.semantic.steps_used, reference.steps_used);
            assert_eq!(actual.semantic.cleanup_trace, reference.cleanup_trace);
        }
    }
}
