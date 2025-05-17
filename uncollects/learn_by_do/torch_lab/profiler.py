import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import time


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate model, loss function, and optimizer
input_size = 100
output_size = 10
model = SimpleModel(input_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create some dummy input data and target labels
batch_size = 64
dummy_input = torch.randn(batch_size, input_size).to(device)
dummy_labels = torch.randint(0, output_size, (batch_size,)).to(device)

# Use the profiler within a 'with' block
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,  # Include CUDA if you are using a GPU
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        "./log/training_step"
    ),  # Optional: for TensorBoard visualization
    record_shapes=True,
    with_stack=True,
) as prof:
    for i in range(5):  # Simulate a few training steps
        # --- Training Step ---
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(dummy_input)  # Forward pass
        loss = criterion(outputs, dummy_labels)  # Calculate loss
        loss.backward()  # Backward pass (calculate gradients)
        optimizer.step()  # Update weights

        # --- Profiler Step ---
        # Signal the profiler to move to the next step.
        # This is crucial when using a schedule.
        prof.step()

# Print the profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
