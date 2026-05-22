#!/usr/bin/env node
"use strict";

const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const repoRoot = path.resolve(__dirname, "..");
const args = process.argv.slice(2);
const command = args[0] || "install";
const installRoot = resolveInstallRoot(args);
const envPython = path.join(installRoot, "envs", "vpa", "bin", "python");

async function main() {
  if (command === "install") {
    if (args.includes("--dry-run")) {
      const dryRun = await ensureDryRunPython();
      try {
        const status = runPythonDryRun(
          dryRun.python,
          ["-m", "vibeproteinagent.install.cli", ...args.slice(1)],
          dryRun.useSourcePath
        );
        if (status !== 0) {
          process.exit(status || 1);
        }
      } finally {
        dryRun.cleanup();
      }
      return;
    }
    await ensureSharedEnv();
    runPython(envPython, ["-m", "vibeproteinagent.install.cli", ...args.slice(1)]);
    return;
  }
  if (command === "doctor") {
    const py = fs.existsSync(envPython) ? envPython : findPython();
    runPython(py, ["-m", "vibeproteinagent.install.cli", "doctor", ...args.slice(1)]);
    return;
  }
  const py = fs.existsSync(envPython) ? envPython : findPython();
  runPython(py, ["-m", "vibeproteinagent.cli.main", ...args]);
}

function resolveInstallRoot(argv) {
  const idx = argv.indexOf("--install-root");
  if (idx >= 0 && argv[idx + 1]) {
    return path.resolve(argv[idx + 1]);
  }
  const eq = argv.find((arg) => arg.startsWith("--install-root="));
  if (eq) {
    return path.resolve(eq.slice("--install-root=".length));
  }
  return path.resolve(process.env.VPA_HOME || path.join(os.homedir(), ".vibeproteinagent"));
}

async function ensureSharedEnv() {
  if (fs.existsSync(envPython)) {
    const existingMicromamba = path.join(installRoot, "bootstrap", "bin", "micromamba");
    if (!process.env.VPA_MICROMAMBA && fs.existsSync(existingMicromamba)) {
      process.env.VPA_MICROMAMBA = existingMicromamba;
    }
    ensurePackageInstalled(envPython);
    return;
  }
  const micromamba = await ensureMicromamba(installRoot);
  process.env.VPA_MICROMAMBA = micromamba;
  fs.mkdirSync(path.dirname(path.dirname(envPython)), { recursive: true });
  runBootstrap(micromamba, [
    "create",
    "-y",
    "-p",
    path.dirname(path.dirname(envPython)),
    "python=3.12",
    "pip"
  ]);
  ensurePackageInstalled(envPython);
}

async function ensureDryRunPython() {
  if (fs.existsSync(envPython)) {
    return { python: envPython, useSourcePath: false, cleanup: () => {} };
  }
  const hostPython = findPython();
  if (pythonCanImportInstallerDeps(hostPython)) {
    return { python: hostPython, useSourcePath: true, cleanup: () => {} };
  }
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), "vpa-dry-run-"));
  const tempPython = path.join(tempRoot, "envs", "vpa", "bin", "python");
  const micromamba = await ensureMicromamba(tempRoot);
  fs.mkdirSync(path.dirname(path.dirname(tempPython)), { recursive: true });
  runBootstrap(micromamba, [
    "create",
    "-y",
    "-p",
    path.dirname(path.dirname(tempPython)),
    "python=3.12",
    "pip"
  ]);
  ensurePackageInstalled(tempPython);
  return {
    python: tempPython,
    useSourcePath: false,
    cleanup: () => fs.rmSync(tempRoot, { recursive: true, force: true })
  };
}

function ensurePackageInstalled(py) {
  const env = {
    ...process.env,
    PYTHONNOUSERSITE: "1",
    PYTHONPATH: "",
    VPA_SOURCE_ROOT: repoRoot
  };
  runBootstrap(py, ["-m", "pip", "install", "--no-user", "-e", repoRoot], env);
}

async function ensureMicromamba(root) {
  if (process.env.VPA_MICROMAMBA && fs.existsSync(process.env.VPA_MICROMAMBA)) {
    return process.env.VPA_MICROMAMBA;
  }
  const onPath = findExecutable("micromamba");
  if (onPath) {
    return onPath;
  }
  const binDir = path.join(root, "bootstrap", "bin");
  const target = path.join(binDir, "micromamba");
  if (fs.existsSync(target)) {
    return target;
  }
  fs.mkdirSync(binDir, { recursive: true });
  const archive = path.join(root, "bootstrap", "micromamba.tar.bz2");
  await download(
    "https://micro.mamba.pm/api/micromamba/linux-64/latest",
    archive
  );
  runBootstrap("tar", ["-xjf", archive, "-C", path.join(root, "bootstrap"), "bin/micromamba"]);
  if (!fs.existsSync(target)) {
    throw new Error(`micromamba bootstrap did not create ${target}`);
  }
  return target;
}

function runPython(py, pyArgs, useSourcePath) {
  const micromamba = process.env.VPA_MICROMAMBA || path.join(installRoot, "bootstrap", "bin", "micromamba");
  const env = {
    ...process.env,
    VPA_HOME: installRoot,
    VPA_SOURCE_ROOT: repoRoot,
    PYTHONNOUSERSITE: "1"
  };
  if (fs.existsSync(micromamba)) {
    env.VPA_MICROMAMBA = micromamba;
  }
  if (useSourcePath) {
    env.PYTHONPATH = path.join(repoRoot, "src");
  }
  run(py, pyArgs, env);
}

function runPythonDryRun(py, pyArgs, useSourcePath) {
  const micromamba = process.env.VPA_MICROMAMBA || path.join(installRoot, "bootstrap", "bin", "micromamba");
  const env = {
    ...process.env,
    VPA_HOME: installRoot,
    VPA_SOURCE_ROOT: repoRoot,
    PYTHONNOUSERSITE: "1"
  };
  if (fs.existsSync(micromamba)) {
    env.VPA_MICROMAMBA = micromamba;
  }
  if (useSourcePath) {
    env.PYTHONPATH = path.join(repoRoot, "src");
  }
  const result = spawnSync(py, pyArgs, {
    stdio: "inherit",
    env
  });
  if (result.error) {
    throw result.error;
  }
  process.exitCode = result.status || 0;
  return result.status || 0;
}

function run(exe, exeArgs, env) {
  const result = spawnSync(exe, exeArgs, {
    stdio: "inherit",
    env: env || process.env
  });
  if (result.error) {
    throw result.error;
  }
  process.exitCode = result.status || 0;
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function runBootstrap(exe, exeArgs, env) {
  const result = spawnSync(exe, exeArgs, {
    text: true,
    encoding: "utf8",
    env: env || process.env
  });
  if (result.stdout) {
    process.stderr.write(result.stdout);
  }
  if (result.stderr) {
    process.stderr.write(result.stderr);
  }
  if (result.error) {
    throw result.error;
  }
  process.exitCode = result.status || 0;
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function findPython() {
  const candidates = [
    process.env.VPA_PYTHON,
    findExecutable("python3"),
    findExecutable("python")
  ].filter(Boolean);
  if (!candidates.length) {
    throw new Error("Python was not found. Run `vibeproteinagent install` first.");
  }
  return candidates[0];
}

function pythonCanImportInstallerDeps(py) {
  const result = spawnSync(
    py,
    ["-c", "import typer, yaml, rich, pydantic"],
    {
      stdio: "ignore",
      env: { ...process.env, PYTHONNOUSERSITE: "1" }
    }
  );
  return result.status === 0;
}

function findExecutable(name) {
  const paths = (process.env.PATH || "").split(path.delimiter);
  for (const dir of paths) {
    const candidate = path.join(dir, name);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

function download(url, destination) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destination);
    const request = https.get(url, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        file.close();
        fs.rmSync(destination, { force: true });
        const nextUrl = new URL(response.headers.location, url).toString();
        download(nextUrl, destination).then(resolve, reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error(`failed to download ${url}: HTTP ${response.statusCode}`));
        return;
      }
      response.pipe(file);
      file.on("finish", () => {
        file.close(resolve);
      });
    });
    request.on("error", reject);
    file.on("error", reject);
  });
}

main()
  .then(() => {
    process.exit(process.exitCode || 0);
  })
  .catch((err) => {
    console.error(err && err.message ? err.message : String(err));
    process.exit(1);
  });
