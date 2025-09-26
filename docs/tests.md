# Testing Guidelines

1. **Command Construction:**
   - Verify that each tool produces the expected command list given a configuration object.
   - Dry-run mode should never raise errors due to missing binaries.

2. **Environment Propagation:**
   - Ensure custom environment variables are forwarded to the subprocess when provided.

3. **Pipeline Sequencing:**
   - Confirm that `GemmaAgent.run_pipeline` executes the four tools in order and passes along the `dry_run` flag.

4. **Error Surfaces:**
   - Simulate command failures by injecting a runner that raises an exception and confirm the error is propagated.

5. **Integration Smoke Test (Optional):**
   - On a GPU-equipped machine with the dependencies installed, run the CLI end-to-end on a small dataset to validate training and conversion.
