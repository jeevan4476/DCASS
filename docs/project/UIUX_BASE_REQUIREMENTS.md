# Phase 3: Stealth AI & Pipeline Integration
## UI/UX Design Baseline Document

> **Purpose:** This document translates the backend engineering accomplished in Phase 3 into functional requirements and data flows specifically targeted for the upcoming Front-End (Next.js) design. It answers the question: *"What backend features do I need to build screens, buttons, and dashboards for?"*

---

## 1. Overview of Phase 3 Accomplishments
In this phase, we completed the **Simulation Pipeline**, effectively connecting Alice (the AI-driven sender) and Bob (the asynchronous receiver) in a containerized environment. 

The major milestone was establishing the **Dynamic→Static Fallback Mechanism**, taking the system from hardcoded scripts to an autonomous pipeline that can gracefully revert to mathematical noise if trained Neural Networks are not available.

---

## 2. Core Screens & Functional Requirements

When designing the UI/UX, you should plan for the following main interfaces:

### A. Alice's Dashboard (Encoding & Sending)
This screen controls the message transmission. The backend currently takes command-line arguments to function; the UI will replace this with visual controls.

*   **Message Input Field**: A standard text area where the user types the secret message.
*   **Stealth Mode Selector (Dropdown/Cards)**:
    *   `Auto (Recommended)`: Let the system attempt RL, then GAN, then Static automatically.
    *   `Static (Fallback)`: Manually force mathematical jitter.
    *   `RL` / `GAN`: Force Neural Network modes.
*   **Behavioral Profile (Dropdown)**: If using Static mode, the user selects their fake identity behavior (`Casual`, `Steady`, `Bursty`, `Night_Owl`).
*   **Base Delay Setting (Slider)**: Controls the baseline speed of transmission (e.g., 3.0 seconds).
*   **Action Button**: `[ ENCODE & TRANSMIT ]` -> Triggers the backend Docker pipeline.

### B. The Network / Wire View (Real-Time Telemetry)
The backend currently outputs `.json` metadata packets to a `shared_channel/` folder. The UI should visualize this folder acting as "The Wire".

*   **Active Transmission Feed**: A real-time scrolling list of packets as they leave Alice and travel the network.
    *   *Data to show per packet:* `Media ID`, `Channel Number`, `Sequence #`, `Delay Applied`.
*   **Mode Indicator**: A badge showing what mode was *actually* used by the backend scheduler (e.g., indicating "Static Fallback Engaged" if no ML model was found).

### C. Bob's Dashboard (Receiving & Decoding)
Bob runs asynchronously in the backend, meaning he waits for silence before acting.

*   **Reassembly Status Bar**: A visual indicator showing Bob's "Silence Threshold" timer counting down (default 10 seconds). Each new packet resets this timer.
*   **Reconstructed Sequence**: Once the timer hits 0, the UI should show the packets locking into their correct `Sequence #` order (since Alice might send them out of order).
*   **Decoded Output (Final Results)**: 
    *   The raw reconstructed semantic string (e.g., `people gathering | park | sunny time`).
    *   A verification accuracy percentage (how many IDs were successfully found in the Faiss text/image databases).

### D. Training Orchestration Panel (Future/Admin View)
We containerized three training scripts. Although training is deferred, the UI should eventually have a panel to trigger them.

*   **Step 1: Traffic Generation**: Button to generate synthetic human data (input: `number of sessions`).
*   **Step 2: Train GAN**: Button to start the Generator vs Warden loop (input: `epochs`).
*   **Step 3: Train RL Agent**: Button to start PPO training (input: `episodes`).
*   **Status Indicators**: The UI can check the `/models/` folder to display a "🟢 Model Ready" or "🔴 Model Missing" status for GAN and RL checkpoints.

---

## 3. Data Hooks for the Next.js Backend

To connect your future Next.js app to this Python backend, your API routes will need to interact with the following resources established in this phase:

1.  **Command Execution (`child_process`)**:
    *   Your API will call `python scripts/docker_orchestrate.py --mode [mode] --send-only` based on UI button clicks.
2.  **File System Watching (`fs.watch`)**:
    *   Your Next.js server can watch the `./shared_channel` directory. Every time a new `media_{id}.json` file appears, send a WebSocket/SSE event to the React frontend to animate a packet arriving!
    *   Watch for `./shared_channel/_manifest.json` to get a summary of the transmission.
3.  **Log Streaming**:
    *   The UI can read from the `./logs/sender` and `./logs/receiver` directories to show the user a high-tech "terminal Output" window on the dashboard.

---

## 4. UI/UX Aesthetic Guidelines
As defined in the project goals, the UI should feel **Premium, Dynamic, and Investigative**. 
*   **Dark Mode First**: Suited for a "stealth" application. Use deep grays/blacks with vibrant, neon accent colors for packets and data streams.
*   **Micro-animations**: Use animations for packets moving from Alice -> Wire -> Bob.
*   **Data Density**: The UI should look technical but not overwhelming, utilizing charts for things like "Jitter distribution" or "Behavioral variation".
