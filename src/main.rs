use candle_core::{Device, Tensor};

fn run_computation(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple tensor on the specified device
    let a = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
    let b = Tensor::new(&[[5f32, 6.], [7., 8.]], &device)?;
    let c = a.matmul(&b)?;

    println!("Matrix multiplication on {:?}:", device);
    println!("A: {:?}", a.to_vec2::<f32>()?);
    println!("B: {:?}", b.to_vec2::<f32>()?);
    println!("A * B = {:?}", c.to_vec2::<f32>()?);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Checking available devices...");

    // Try to use Metal (GPU acceleration on macOS)
    match Device::new_metal(0) {
        Ok(device @ Device::Metal(_)) => {
            println!("✅ Metal GPU device available: {:?}", device);
            run_computation(&device)?;
        }
        Ok(device) => {
            println!("⚠️  Expected Metal device but got: {:?}", device);
            println!("Falling back to CPU");
            let device = Device::Cpu;
            run_computation(&device)?;
        }
        Err(_) => {
            println!("⚠️  Metal GPU not available, falling back to CPU");
            let device = Device::Cpu;
            run_computation(&device)?;
        }
    }

    Ok(())
}
