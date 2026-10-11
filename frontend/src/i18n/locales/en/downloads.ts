import type zh from "../zh-TW/downloads.ts";
import type { Shape } from "../../types.ts";

const downloads: Shape<typeof zh> = {
  title: "Downloads",
  description:
    "Fetch models from Hugging Face; finished ones appear in the Models library.",
  unsupportedTitle: "This engine version has no download manager",
  unsupportedDescription:
    "Upgrade the engine to download models here; for now use Import model in the Models library.",
  "add.title": "Add a download",
  "add.description":
    "Enter a repository; leave the revision and patterns empty to get the whole model.",
  "add.repo": "Repository",
  "add.repoInvalid": "Use the form org/name.",
  "add.revision": "Revision (branch or tag)",
  "add.patterns": "File patterns",
  "add.patternsHint":
    "Comma-separated, for example only the weights and config; empty means every file.",
  "add.start": "Start download",
  "add.checking": "Checking space",
  "add.needRepo": "Enter a valid repository first.",
  "add.free": "Free space for models",
  "disk.title": "Not enough disk space",
  "disk.body":
    "Needs {needed} but only {free} is free. Free some space or choose a smaller quantization.",
  details: "Details",
  offlineReason: "The engine is offline; downloads are unavailable.",
  busyReason: "Another download is being submitted.",
  "state.queued": "Queued",
  "state.running": "Downloading",
  "state.done": "Done",
  "state.failed": "Failed",
  "state.cancelled": "Cancelled",
  cancel: "Cancel",
  resume: "Resume",
  openModel: "Open model",
  progressAria: "Download progress for {repo}",
  rateTitle: "Average speed over the last seconds",
  eta: "{time} left",
  files: "{done} / {total} files",
  alreadyPresent: "The model is already on disk; nothing was downloaded.",
  presentNow: "{repo} is already on disk; nothing to download.",
  notRegistered:
    "Downloaded but not registered yet; it is picked up after an engine restart.",
  failedSummary: "Download failed, see why",
  emptyTitle: "No downloads yet",
  emptyDescription: "Downloads you add show their progress here.",
  loadingTitle: "Loading",
  dir: "Models directory",
  pollError: "Download status is unavailable right now; showing the last data.",
};

export default downloads;
