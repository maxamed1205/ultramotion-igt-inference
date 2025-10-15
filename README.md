# ultramotion-igt-inference

Prototype repository for a deployable OpenIGTLink inference service

Goal
----
This repository contains a minimal, container-friendly skeleton for a real-time
inference service that subscribes to an OpenIGTLink image stream (PlusServer),
runs a segmentation pipeline (D-FINE + MobileSAM) and republishes a binary
labelmap (`BoneMask`) back to 3D Slicer via OpenIGTLink.

Status
------
This is an initial scaffold: skeleton service, Dockerfile placeholder and CI
workflow. Implementation of the real models and heavy GPU testing will be
completed later.

Quick start (dev)
------------------
1. Build the container (requires NVIDIA Docker on Linux / WSL2):

   # build command will be provided in the Dockerfile section

2. Start your PlusServer and Slicer clients. Configure device names:
   - IMAGE: `Image_Ultrasound`
   - TRANSFORM: `ImageToReference`
   - MASK OUT: `BoneMask`

3. Run the service and verify it connects to PlusServer and opens a server
   endpoint for Slicer.

See `docker/` and `src/` for the service skeleton.

License
-------
MIT
