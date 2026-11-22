import doneJson from "./contract/job_done.example.json";
import runningJson from "./contract/job_running.example.json";
import type { JobResponse } from "./types";

// JSON imports widen enum fields to `string`, so `satisfies` cannot be used directly;
// shape is verified at runtime in lib/mock.test.ts instead.
export const mockDone = doneJson as unknown as JobResponse;
export const mockRunning = runningJson as unknown as JobResponse;
