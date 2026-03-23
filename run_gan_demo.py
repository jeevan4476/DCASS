import sys
import time
import torch

sys.path.append("src/stealth/gan")
from generator import TemporalPatternGenerator

MESSAGE = "a dog running through water and a man riding a bicycle"

semantic_payload = [
    "img_4521.jpg",
    "img_982.jpg"
]

CHANNELS = {
    0: "WhatsApp",
    1: "Instagram",
    2: "Email"
}

print("\nDCASS GAN STEALTH DEMO")

device = "cpu"

generator = TemporalPatternGenerator(
    latent_dim=128,
    hidden_dim=256,
    num_channels=3,
    max_sequence_length=100
).to(device)

generator.eval()

schedule = generator.generate(
    batch_size=1,
    sequence_length=len(semantic_payload),
    time_of_day=torch.tensor([14.0]),
    device=device
)

delays = schedule.delays[0]
channels = schedule.sample_channels()[0]

print("\nGenerated Schedule:\n")

for i in range(len(semantic_payload)):
    d = delays[i].item()
    ch = CHANNELS[channels[i].item()]

    print(f"Step {i+1} → Delay {d:.2f}s → Channel {ch}")

print("\nSimulated Transmission\n")

for i, payload in enumerate(semantic_payload):

    d = delays[i].item()
    ch = CHANNELS[channels[i].item()]

    print(f"Waiting {d:.2f}s")
    time.sleep(min(d,2))

    print(f"Sending {payload} via {ch}")

print("\nReceiver reconstructed message:")
print("dog playing in water. man riding bicycle")