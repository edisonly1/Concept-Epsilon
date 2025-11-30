# Concept Epsilon – Hydraulic Transient Forensics for Water Grid Resilience

Municipal water systems routinely lose 20–30% of treated water to leaks and bursts, but most
utilities only discover failures when someone sees water in the street. Conventional leak detection
either sends crews walking the network with acoustic listening sticks or requires dense, expensive
sensor deployments.

**Concept Epsilon** is a software only “Transient Forensics Engine” that uses existing pumps,
valves, and SCADA pressure sensors to localize leaks:

- Use **routine pump/valve operations** as deliberate “pings” of the network.
- Use a **transient hydraulic solver (TSNet)** to simulate how pressure waves propagate.
- Train a **1D CNN** on synthetic leak scenarios to recognize where the leak likely is.
- Add a **physics-consistency check** that resimulates the predicted leak and rejects
  ML guesses that contradict the hydraulics.
- Present results in a lightweight **web dashboard** that highlights likely leak locations
  and shows observed vs simulated pressure traces.

<img width="1897" height="956" alt="luma" src="https://github.com/user-attachments/assets/8fd14f62-75e2-4ea6-82ad-54f7ecbff10b" />

---

## 1. Idea

Concept Epsilon turns daily operations (pump stops, valve closures) into a leak scan of the entire
network. When a pump trips or a valve closes, a high-speed pressure wave travels through the
pipes; leaks change how that wave reflects and decays. We simulate these transients with TSNet
on a digital twin of the network, generate a synthetic library of “what the wave looks like if the
leak is on pipe X,” and train a 1D CNN to map real pressure traces back to a leak location. A
physics-consistency layer then resimulates the top ML candidates and only reports leak locations
whose simulated transients closely match the observed one.

---

## 2. How it works (architecture + science)

Concept Epsilon is built around **inverse transient analysis**: pressure waves in a pressurized pipe
obey well-known physics, and leaks change how those waves propagate, reflect, and decay. The
system uses TSNet as a forward solver for the 1D **water hammer equations** and a compact
neural network to approximate the inverse mapping from pressure traces back to leak location.

It has five main pieces:

1. **Network Model (TSNet + EPANET)**  
   - We start from an EPANET `.inp` file that encodes the network topology (nodes, pipes, pumps,
     valves), base demands, and boundary conditions.  
   - TSNet wraps this EPANET model and solves the 1D unsteady flow equations for head `H(x, t)`
     and discharge `Q(x, t)` along each pipe. These are the standard water hammer PDEs that
     enforce conservation of mass and momentum with compressible water and elastic pipes.  
   - Numerically, TSNet uses a **Method-of-Characteristics (MOC)** finite-difference scheme:
     it discretizes each pipe into segments, marches forward in time, and updates `H` and `Q`
     along characteristic lines. This captures:
     - Wave propagation at the acoustic speed `a ≈ 1000–1200 m/s`,
     - Reflections at junctions, dead-ends, and diameter changes,
     - Damping due to wall friction and any energy “leaks” out of the system.  
   - We define a repeatable **excitation event** (e.g., a rapid valve closure or pump trip) that acts
     like an active “ping” of the network. When the actuator moves, it launches a pressure wave;
     that wave’s timing and amplitude at a few sensor nodes become our observable data.

2. **Synthetic Leak Library**  
   - Physically, a leak behaves like an additional outflow term in the continuity equation: flow is
     lost to the environment, which locally reduces pressure and changes how waves reflect. In
     EPANET/TSNet we model this as an **emitter** or “leak node” with
     `Q_leak = C_e * H^m`, approximating orifice flow through a small opening.  
   - For each candidate pipe (or its associated node) and for a range of leak sizes we:
     - Insert a leak into the TSNet model,
     - Run the same excitation event (same pump/valve schedule, same total time `T`, same
       time step `Δt`),
     - Record high-frequency pressure/head traces `H_s^(p)(t_k)` at a small set of “SCADA
       sensors” `s` in a sensor set `S`.  
   - This yields a **supervised synthetic dataset**:
     - Data set `D = { (x^(i), y^(i)) }` for `i = 1, ..., N`, where  
       - `x^(i)` is a multichannel time series built from pressure at one or more sensors
         (possibly including derived channels like `ΔH/Δt`),  
       - `y^(i)` is the leak label (“no leak” or a pipe ID).  
   - We align all traces to the start of the excitation, optionally resample to reduce dimensionality,
     normalize each channel, and can add small noise to approximate realistic sensor behavior.

3. **Leak Locator Model (1D CNN)**  
   - The scientific signal we care about is encoded in the **waveform**: when specific wave
     fronts arrive at each sensor, how they are reflected, and how fast their amplitude decays.
     Different leak locations produce different wave “fingerprints.”  
   - Instead of hand-crafting features (arrival times, decay rates), we use a 1D convolutional neural
     network `f_theta` that learns these patterns directly from the time series:
     - Input: `x` in `R^(T × C)` (T time steps, C channels = sensors × features),  
     - Output: `p_hat = softmax(f_theta(x))` in `R^(|P|)`, a probability distribution over pipes
       (plus a “no leak” class).  
   - The CNN uses temporal convolutions to capture local wavefront shapes, pooling to build
     invariance to small timing shifts, and a dense head to map those learned patterns to leak
     classes.  
   - Training:
     - Train/validation split over synthetic scenarios,
     - Categorical cross-entropy loss and Adam optimizer,
     - Early stopping on validation loss to avoid overfitting to specific leak magnitudes or noise
       realizations.  
   - In effect, this is a data-driven approximation to the **inverse transient problem**: given
     pressure vs time at a few points and a known excitation, infer which change in network
     impedance (which leak) best explains the observation.

4. **Physics-Consistency Check**  
   - Machine learning alone can misclassify in ways that violate the hydraulics (e.g., picking a pipe
     whose leak would never produce the observed waveform). To avoid this, we wrap the CNN in a
     **forward-physics verification** step.  
   - Treat TSNet as a forward model `F(theta)` that maps a set of network parameters
     `theta` (including leak location and size) to simulated sensor traces. For each of the top `k`
     CNN candidates:
     1. Construct `theta_p` corresponding to “leak at pipe p with reasonable size”,  
     2. Run TSNet to obtain simulated traces `x_sim^(p) = F(theta_p)`,  
     3. Compare to the observed trace `x_obs` using a discrepancy metric, e.g.,
        normalized mean-squared error or correlation. For example:
        - `E(p) = (1/T) * sum_t (x_obs(t) - x_sim^(p)(t))^2`.  
   - We then compute a **fused score**:
     - `combined_score(p) = ML_prob(p) * physics_similarity(p)`  
     and re-rank candidates by this value.  
   - Only leak locations that are both **high-probability under the learned model** and **produce
     a TSNet transient that matches the observed data** are reported, making the system more
     trustworthy under distribution shift and model uncertainty.

5. **“Pulse” Dashboard (Web Frontend)**  
   - A lightweight React/Vite/Tailwind frontend acts as the **operator console** for this physics–ML
     engine. It is designed to make the underlying science visible:  
     - A stylized map shows nodes and pipes; the predicted leak pipe is highlighted, turning the
       abstract inverse problem into a concrete asset on the screen.  
     - A time-series panel plots the **observed pressure trace** and the **TSNet-predicted trace
       for the best leak candidate**; when the curves align, users see that the candidate is not
       just a neural net guess but is physically plausible.  
     - A result card exposes:
       - The predicted leak location (pipe ID and approximate position),
       - The CNN confidence,
       - The physics similarity score (e.g., correlation),  
       giving engineers a transparent breakdown of “what the model thinks” versus
       “what the physics says.”  
   - In a real deployment, this dashboard would sit on top of SCADA: utilities would schedule a
     controlled pump/valve operation, stream the resulting high-frequency pressure data to the
     engine, and use the dashboard to triage likely leak locations for field crews.

---

## 3. How to run

### 3.1 Download and open in VS Code

1. Go to the GitHub repository and click **Code → Download ZIP**.  
2. Unzip the downloaded file.  
3. Open the unzipped folder in **VS Code**.  
   You should see two main folders:
   - `backend/`
   - `frontend/`

---

### 3.2 Backend – API + simulation (Python)

1. **Create and activate a virtual environment**

   In VS Code, open a terminal and run:

   ```bash

   # Create virtual environment
   python3.11 -m venv .venv

   # Activate venv (macOS / Linux)
   source .venv/bin/activate

   # or on Windows (PowerShell)
   # .venv\Scripts\Activate
2. Install Python dependencies

With the virtual environment active:

  ```bash
  #install requirements
  pip install -r requirements.txt
  #Start the backend server:
  cd backend
  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
  #Verify the backend is running
```
In a browser or terminal:
```bash
curl http://127.0.0.1:8000/api/health
```
Expected response:
{"status": "ok"}
Keep this terminal open while you use the app.

3.3 Frontend – Web dashboard (React)
Open a new terminal in VS Code.

Navigate to the frontend folder:
```bash
cd frontend
#Install Node.js dependencies:
npm install
#Start the development server:
npm run dev
#Open the URL printed in the terminal (usually http://localhost:5173) in your browser.
```

The frontend is configured to talk to the backend at http://127.0.0.1:8000/api, so make sure
the backend is running before you interact with the UI.



