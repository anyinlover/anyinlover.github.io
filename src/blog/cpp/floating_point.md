---
title: Floating-Point Representation: A Deep Dive
pubDate: 2025-03-14 14:43:00
tags:
- float, precision
---

# Floating-Point Representation: A Deep Dive

## IEEE 754 Floating-Point Standard

The IEEE 754 standard defines how floating-point numbers are represented in computers. A number, $V$, is expressed as:

$$
V = (-1)^s \times M \times 2^E
$$

Where:

* **$s$ (Sign):** Determines the sign of the number: $s = 0$ for positive, $s = 1$ for negative.
* **$M$ (Significand/Mantissa):** A fractional binary number. It ranges from $1$ to $2 - \epsilon$ for normalized values, or from $0$ to $1 - \epsilon$ for denormalized values, where $\epsilon$ is the machine epsilon.
* **$E$ (Exponent):** Weights the value by a power of 2.

### Common Floating-Point Formats

The following table summarizes the key characteristics of common floating-point formats:

| Format | Total Bits | Exponent Bits ($k$) | Fraction Bits ($n$) |
| :----- | :--------- | :------------------ | :------------------ |
| Double | 64         | 11                  | 52                  |
| Float  | 32         | 8                   | 23                  |
| FP16   | 16         | 5                   | 10                  |
| BF16   | 16         | 8                   | 7                   |

### Special Value Categories

Floating-point numbers can represent various special values, defined by the exponent ($e$) and fraction ($f$) fields:

| Category            | Condition                 | Value                                       |
| :------------------ | :------------------------ | :------------------------------------------ |
| Normalized Values   | $0 < e < 2^k - 1$         | $(-1)^s \times (1 + f) \times 2^{e - bias}$ |
| Denormalized Values | $e = 0$                   | $(-1)^s \times f \times 2^{1 - bias}$       |
| Infinity            | $e = 2^k - 1$, $f = 0$    | $(-1)^s \times \infty$                      |
| NaN (Not a Number)  | $e = 2^k - 1$, $f \neq 0$ | NaN                                         |

Where the bias is $2^{k-1} - 1$.

**Denormalized numbers** serve two crucial purposes:

1.  **Representation of Zero:** They allow for distinct representations of positive ($+0.0$) and negative ($-0.0$) zero, differentiated by the sign bit.
2.  **Representation of Values Close to Zero:** They enable the representation of numbers very close to $0.0$, filling the gap between zero and the smallest normalized number.

### Example: 6-Bit Floating-Point Format

Let's illustrate with a 6-bit floating-point format (1 sign bit, 3 exponent bits, 2 fraction bits):

| Description          | Bit Representation | $E$  | $f$  | Value    |
| :------------------- | :----------------- | :--- | :--- | :------- |
| Zero                 | 0 000 00           | -3   | 0/4  | 0/32     |
| Smallest Positive    | 0 000 01           | -3   | 1/4  | 1/32     |
| Largest Denormalized | 0 000 11           | -3   | 3/4  | 3/32     |
| Smallest Normalized  | 0 001 00           | -2   | 0/4  | 4/32     |
| One                  | 0 011 00           | 0    | 0/4  | 4/4      |
| Largest Normalized   | 0 110 11           | 3    | 3/4  | 28/4     |
| Infinity             | 0 111 00           | -    | -    | $\infty$ |

## Rounding Modes

When a number cannot be represented exactly, rounding is necessary. Common rounding modes include:

| Mode               | 1.4  | 1.6  | 1.5  | 2.5  | -1.5 |
| :----------------- | :--- | :--- | :--- | :--- | :--- |
| Round-to-Even      | 1    | 2    | 2    | 2    | -2   |
| Round-Toward-Zero  | 1    | 1    | 1    | 2    | -1   |
| Round-Down (Floor) | 1    | 1    | 1    | 2    | -2   |
| Round-Up (Ceiling) | 2    | 2    | 2    | 3    | -1   |

## Floating-Point Operations: Precision and Pitfalls

A significant challenge with floating-point arithmetic is the "big eats small" phenomenon. Consider:

$$
3.14 + 1 \times 10^{10} - 1 \times 10^{10} = 0.0
$$

This occurs due to the following steps in floating-point addition:

1.  **Alignment:** Exponents are aligned by shifting the significand of the smaller number until both exponents match.
2.  **Significand Addition:** The significands are added.
3.  **Normalization and Rounding:** The result is normalized and rounded if necessary.

Precision loss happens during the alignment step when one number is significantly larger than the other.

### Python Representation of Floating-Point Numbers

The following Python code demonstrates how to decompose a float into its significand and exponent:

```python
import struct

def float_to_fe(f):
    packed = struct.pack('>f', f)
    int_val = struct.unpack('>I', packed)[0]
    sign = (int_val >> 31) & 1
    exponent = (int_val >> 23) & 0xFF
    mantissa = int_val & 0x7FFFFF

    if exponent == 0xFF:  # Infinity or NaN
        if mantissa == 0:
            return "Infinity" if sign == 0 else "-Infinity"
        else:
            return "NaN"

    bias = 127
    if exponent == 0:
        e = 1 - bias
        mantissa_binary = f"0.{mantissa:023b}" #denormalized
    else:
        e = exponent - bias
        mantissa_binary = f"1.{mantissa:023b}" #normalized

    if sign == 1:
        mantissa_binary = "-" + mantissa_binary

    return f"{mantissa_binary} * 2^{e}"
```

Example:

$$
3.14 = 1.10010001111010111000011 \times 2^1 \\
1 \times 10^{10} = 1.00101010000001011111001 \times 2^{33}
$$

To align $3.14$ with $1 \times 10^{10}$, its significand must be right-shifted by 32 bits. Due to the 23-bit fraction field in single-precision floats, $3.14$ effectively becomes $0.0$.

## Conversion Between Floating-Point Formats

Converting between floating-point formats can lead to:

* **Overflow:** If the target format's exponent range is smaller.
* **Loss of Precision/Underflow:** If the target format's fraction field is smaller.
