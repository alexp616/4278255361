using CUDA
using BenchmarkTools

include("4278255361.jl")

function cpu_benchmark()
    println("CPU Benchmarks: ")
    trials = 1000000000

    println("Reducing $trials numbers using special method:")
    @time begin
        for i in 1:1000000000
            x = rand(UInt64)
            a = reduce_mod_p(x)
        end
    end

    println("Reducing $trials numbers using normal method:")
    @time begin
        for i in 1:1000000000
            x = rand(UInt64)
            a = UInt32(x % p_64)
        end
    end
    println()
end

function gpu_benchmark()
    println("GPU Benchmarks: ")
    len = 2^25 # 33,554,432
    trials = 1000

    A = CUDA.rand(UInt64, len)
    B = CUDA.zeros(UInt32, len)

    kernel1 = @cuda launch=false reduce_kernel_1(A, B, trials)
    config = launch_configuration(kernel1.fun)
    threads = min(len, Base._prevpow2(config.threads))
    blocks = cld(len, threads)

    kernel1(A, B, trials; threads = threads, blocks = blocks)
    println("Reducing $(len * trials) numbers using special method")
    CUDA.@time kernel1(A, B, trials; threads = threads, blocks = blocks)

    kernel2 = @cuda launch=false reduce_kernel_2(A, B, trials)
    config = launch_configuration(kernel2.fun)
    threads = min(len, Base._prevpow2(config.threads))
    blocks = cld(len, threads)

    kernel2(A, B, trials; threads = threads, blocks = blocks)
    println("Reducing $(len * trials) numbers using normal method")
    CUDA.@time kernel2(A, B, trials; threads = threads, blocks = blocks)
end

function reduce_kernel_1(A, B, trials)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds for i in 1:trials
        B[tid] = reduce_mod_p(A[tid])
    end
end

function reduce_kernel_2(A, B, trials)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds for i in 1:trials
        B[tid] = UInt32(A[tid] % p_64)
    end
end

function cputest()
    for i in 1:100000000
        x = rand(UInt64)
        @assert UInt32(x % p_64) == reduce_mod_p(x)
    end

    for i in 1:100000000
        x = rand(UInt32)
        y = rand(UInt32)
        @assert UInt32(widemul(x, y) % p_64) == mul_mod_p(x, y)
    end
end

function view_llvm()
    x = rand(UInt64)
    @code_native debuginfo = :none reduce_mod_p(x)
end