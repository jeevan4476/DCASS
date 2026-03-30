"""
DCASS Demo Web Server
Runs the semantic steganography + GAN transmission demo with a live dashboard.
"""

import sys
import time
import json
import re
import threading
import queue
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, Response, stream_with_context
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.stealth.gan.generator import TemporalPatternGenerator

app = Flask(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────

CHANNELS = {0: "SocialApp-Alpha", 1: "SocialApp-Beta", 2: "SocialApp-Gamma"}

DEMO_SCENARIOS = [
    {
        "id": "scenario_1",
        "label": "Scenario A",
        "message": "a dog running through water and a man riding a bicycle",
        "semantic_payload": [
            {"chunk": "a dog running through water", "image_id": "img_4521.jpg", "decoded": "dog playing in water"},
            {"chunk": "a man riding a bicycle",      "image_id": "img_0982.jpg", "decoded": "man cycling down road"},
        ],
    },
    {
        "id": "scenario_2",
        "label": "Scenario B",
        "message": "a cat sleeping on a sofa and children playing in a park",
        "semantic_payload": [
            {"chunk": "a cat sleeping on a sofa",    "image_id": "img_1103.jpg", "decoded": "cat resting on furniture"},
            {"chunk": "children playing in a park",  "image_id": "img_2247.jpg", "decoded": "kids at outdoor play area"},
            {"chunk": "sunny afternoon",             "image_id": "img_3391.jpg", "decoded": "bright daytime scene"},
        ],
    },
    {
        "id": "scenario_3",
        "label": "Scenario C",
        "message": "a mountain peak covered in snow and a river flowing through a valley",
        "semantic_payload": [
            {"chunk": "a mountain peak covered in snow", "image_id": "img_5512.jpg", "decoded": "snow-capped mountain summit"},
            {"chunk": "a river flowing through a valley","image_id": "img_6734.jpg", "decoded": "river winding through lowland"},
        ],
    },
]


# ─── GAN Setup ───────────────────────────────────────────────────────────────

device = "cpu"
generator = TemporalPatternGenerator(
    latent_dim=128, hidden_dim=256, num_channels=3, max_sequence_length=100
).to(device)
generator.eval()


def run_gan_schedule(scenario: dict, hour: float) -> dict:
    """Run GAN to produce a schedule for the given scenario."""
    n = len(scenario["semantic_payload"])
    with torch.no_grad():
        schedule = generator.generate(
            batch_size=1,
            sequence_length=n,
            time_of_day=torch.tensor([hour]),
            device=device,
        )
    delays   = schedule.delays[0].tolist()
    channels = schedule.sample_channels()[0].tolist()
    confidence = schedule.confidence[0].item()
    return {"delays": delays, "channels": channels, "confidence": confidence}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", scenarios=DEMO_SCENARIOS)


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify(DEMO_SCENARIOS)


@app.route("/api/run/<scenario_id>")
def api_run(scenario_id):
    scenario = next((s for s in DEMO_SCENARIOS if s["id"] == scenario_id), None)
    if not scenario:
        return jsonify({"error": "scenario not found"}), 404

    hour = datetime.now().hour + datetime.now().minute / 60
    gan_out = run_gan_schedule(scenario, hour)
    payload = scenario["semantic_payload"]

    steps = []
    for i, item in enumerate(payload):
        steps.append({
            "step": i + 1,
            "chunk": item["chunk"],
            "image_id": item["image_id"],
            "decoded": item["decoded"],
            "delay_s": round(gan_out["delays"][i], 3),
            "channel": CHANNELS[int(gan_out["channels"][i])],
            "channel_id": int(gan_out["channels"][i]),
            "timestamp": None,  # filled by frontend
        })

    return jsonify({
        "scenario": scenario,
        "steps": steps,
        "confidence": round(gan_out["confidence"], 4),
        "hour": round(hour, 2),
        "total_delay": round(sum(gan_out["delays"]), 2),
    })


@app.route("/api/stream/<scenario_id>")
def api_stream(scenario_id):
    """SSE stream that emits each transmission step in real time."""
    scenario = next((s for s in DEMO_SCENARIOS if s["id"] == scenario_id), None)
    if not scenario:
        return jsonify({"error": "not found"}), 404

    hour = datetime.now().hour + datetime.now().minute / 60
    gan_out = run_gan_schedule(scenario, hour)
    payload  = scenario["semantic_payload"]

    def generate():
        # Phase 1 – GAN schedule ready
        yield f"data: {json.dumps({'phase': 'schedule', 'confidence': round(gan_out['confidence'], 4), 'hour': round(hour, 2)})}\n\n"
        time.sleep(0.4)

        # Phase 2 – simulate each transmission
        for i, item in enumerate(payload):
            raw_delay = gan_out["delays"][i]
            capped = min(raw_delay, 3.0)   # cap sleep for demo

            yield f"data: {json.dumps({'phase': 'waiting', 'step': i+1, 'delay': round(raw_delay, 3)})}\n\n"
            time.sleep(capped)

            step_data = {
                "phase":   "transmit",
                "step":    i + 1,
                "total":   len(payload),
                "chunk":   item["chunk"],
                "image_id":item["image_id"],
                "channel": CHANNELS[int(gan_out["channels"][i])],
                "channel_id": int(gan_out["channels"][i]),
                "delay":   round(raw_delay, 3),
            }
            yield f"data: {json.dumps(step_data)}\n\n"
            time.sleep(0.2)

        # Phase 3 – decode
        time.sleep(0.5)
        decoded = [item["decoded"] for item in payload]
        reconstructed = ". ".join(decoded) + "."
        yield f"data: {json.dumps({'phase': 'decoded', 'captions': decoded, 'reconstructed': reconstructed})}\n\n"

        yield f"data: {json.dumps({'phase': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
