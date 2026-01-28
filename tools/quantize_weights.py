import numpy as np

# Layer sizes
INPUT_DIM = 784
HIDDEN_DIM = 128
HIDDEN_DIM2 = 64
OUTPUT_DIM = 10

sizes = [
    HIDDEN_DIM * INPUT_DIM,   # weights1
    HIDDEN_DIM,               # bias1
    HIDDEN_DIM2 * HIDDEN_DIM, # weights2
    HIDDEN_DIM2,              # bias2
    OUTPUT_DIM * HIDDEN_DIM2, # weights3
    OUTPUT_DIM                # bias3
]

names = [
    "W1", "B1",
    "W2", "B2",
    "W3", "B3"
]

# Read dump
with open("weights_dump.txt", "r") as f:
    raw = f.read()

# Extract floats
values = [float(x.replace("f", "")) for x in raw.replace("\n", "").split(",") if x]


assert sum(sizes) == len(values), "Size mismatch!"

layers = {}
idx = 0
for name, size in zip(names, sizes):
    layers[name] = np.array(values[idx:idx+size], dtype=np.float32)
    idx += size

# Quantize weights (bias stays int32)
def quantize(arr):
    max_val = np.max(np.abs(arr))
    scale = max_val / 127.0 if max_val != 0 else 1e-6
    q = np.round(arr / scale).astype(np.int8)
    return q, scale

# Write header
with open("src_int/trained_weights.h", "w") as f:
    f.write("#ifndef TRAINED_WEIGHTS_H\n")
    f.write("#define TRAINED_WEIGHTS_H\n\n")
    f.write("#include <stdint.h>\n\n")

    input_scale = 1.0 / 255.0
    
    # Layer 1
    w1_q, w1_scale = quantize(layers["W1"])
    acc1_scale = input_scale * w1_scale
    b1_q = np.round(layers["B1"] / acc1_scale).astype(np.int32)
    
    # Layer 2
    act1_scale = acc1_scale * 400.0 # Heuristic output scale
    w2_q, w2_scale = quantize(layers["W2"])
    acc2_scale = act1_scale * w2_scale
    b2_q = np.round(layers["B2"] / acc2_scale).astype(np.int32)

    # Layer 3
    act2_scale = acc2_scale * 400.0 
    w3_q, w3_scale = quantize(layers["W3"])
    acc3_scale = act2_scale * w3_scale
    b3_q = np.round(layers["B3"] / acc3_scale).astype(np.int32)

    # Write W1
    f.write(f"#define W1_SCALE {w1_scale}f\n")
    f.write("static const int8_t W1_Q[] = {\n")
    f.write(", ".join(map(str, w1_q.tolist())))
    f.write("\n};\n\n")

    # Write W2
    f.write(f"#define W2_SCALE {w2_scale}f\n")
    f.write("static const int8_t W2_Q[] = {\n")
    f.write(", ".join(map(str, w2_q.tolist())))
    f.write("\n};\n\n")

    # Write W3
    f.write(f"#define W3_SCALE {w3_scale}f\n")
    f.write("static const int8_t W3_Q[] = {\n")
    f.write(", ".join(map(str, w3_q.tolist())))
    f.write("\n};\n\n")

    # Write B1
    f.write("static const int32_t B1_Q[] = {\n")
    f.write(", ".join(map(str, b1_q.tolist())))
    f.write("\n};\n\n")

    # Write B2
    f.write("static const int32_t B2_Q[] = {\n")
    f.write(", ".join(map(str, b2_q.tolist())))
    f.write("\n};\n\n")

    # Write B3
    f.write("static const int32_t B3_Q[] = {\n")
    f.write(", ".join(map(str, b3_q.tolist())))
    f.write("\n};\n\n")

    f.write("#define INPUT_SCALE (1.0f / 255.0f)\n")
    f.write(f"#define ACT1_SCALE (INPUT_SCALE * W1_SCALE * 400.0f)\n")
    f.write(f"#define ACT2_SCALE (ACT1_SCALE * W2_SCALE * 400.0f)\n")
    f.write(f"#define OUTPUT_SCALE (ACT2_SCALE * W3_SCALE * 400.0f)\n")

    f.write("#endif\n")

print("✅ trained_weights.h generated successfully")
