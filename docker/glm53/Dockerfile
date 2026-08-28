FROM vllm/vllm-openai:glm53-flash-arm64-cu130

# Spark kernel layer (local tag glm53-flash-sm121:v8).
# This 2× GB10 kit serves MM + Ray on :8888 from the mm-ray tag on top.
#
# Stock glm53-flash-arm64-cu130 sends SM12 to FLASHINFER_MLA_SPARSE_SM120
# (packed DeepSeek cache, pe_dim=64). Our LibertAIDAI checkpoint is NoPE
# (pe_dim=0), so we pin FI 0.6.18, put NCCL / cutlass-dsl back, then run
# glm53-flash_SM121.py with default sm90. Do not pass sm120 — that packed path
# is unused. Build context is files/ (COPY glm53-flash_SM121.py beside this file).

# FlashInfer 0.6.18 for this image (SM90 NoPE MLA + FA2 on GB10).
# Drop flashinfer-jit-cache so this kit does not load SM120 cubins.
RUN pip install -q --pre \
      flashinfer-python==0.6.18.dev20260819 \
      flashinfer-cubin==0.6.18.dev20260819 \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && python3 -c "import flashinfer; print('[mia-sm121] flashinfer', flashinfer.__version__)"

# FI nightly pulls an older nvidia-nccl-cu13. This kit's CX7 needs 2.30.7.
RUN pip install -q nvidia-nccl-cu13==2.30.7 \
    && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" \
    && echo '[mia-sm121] nvidia-nccl-cu13==2.30.7 (CX7 on this kit)'

# Same nightly skews nvidia-cutlass-dsl; GB10 CuTeDSL wants 4.6.2 back.
RUN pip install -q nvidia-cutlass-dsl==4.6.2 \
    && pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2" \
    && echo '[mia-sm121] nvidia-cutlass-dsl==4.6.2 (GB10 CuTeDSL)'

# vLLM + FI trees are both on disk here. Default sm90 (seven steps).
COPY glm53-flash_SM121.py /tmp/glm53-flash_SM121.py
RUN python3 /tmp/glm53-flash_SM121.py && rm /tmp/glm53-flash_SM121.py
