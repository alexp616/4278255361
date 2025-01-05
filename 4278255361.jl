const p = 0xff00ff01
const p_64 = UInt64(p)

function reduce_mod_p(x::UInt32)::UInt32
    return x >= p ? x - p : x
end

function reduce_mod_p(x::UInt64)::UInt32
    # high = first 24 bits
    # mid = next 8 bits
    # low = last 32 bits
    # x = high * 2^40 + mid * 2^32 + low
    high = UInt32(x >> 40)
    mid = UInt32((x >> 32) & 0x000000ff)
    low = UInt32(x & 0xffffffff)

    # 2^40 = -1 mod p, so
    # result = (low - high) + mid * 2^32 mod p
    lowminushigh = low - high

    if high > low
        lowminushigh += p
    end

    # 2^32 = 2^24 - 2^16 + 2^8 - 1 mod p, so
    # result = (low - high) + mid*(2^24 - 2^16 + 2^8 - 1) mod p
    product = (mid << 24) - (mid << 16) + (mid << 8) - mid

    # same as add_mod_p(low2, product) from here on
    result = lowminushigh + product

    if (result < product) || (result >= p)
        result -= p
    end

    return result
end

function mul_mod_p(x::UInt32, y::UInt32)::UInt32
    prod = widemul(x, y)

    return reduce_mod_p(prod)
end

function add_mod_p(x::UInt32, y::UInt32)::UInt32
    sum = x + y
    if (sum < x) || (sum > p)
        sum -= p
    end
    return sum
end

function sub_mod_p(x::UInt32, y::UInt32)::UInt32
    if y > x
        return (p - y) + x
    else
        return x - y
    end
end


