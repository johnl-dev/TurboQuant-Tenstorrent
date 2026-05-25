// host/turboquant_launcher.cpp
//
// Interactive launcher for turboquant_host.
// Prompts the user for configuration options (with sensible defaults) then
// execs turboquant_host with the resulting arguments.
//
// Mirrors the structure of rref_tenstorrent/rref_launcher.cpp.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

static constexpr const char* HOST_EXE = "./turboquant_host";

// ---------------------------------------------------------------------------
// LaunchConfig
// ---------------------------------------------------------------------------

struct LaunchConfig {
    uint32_t    num_vectors = 64;
    std::string mode        = "full";   // "full" only for now
};

// ---------------------------------------------------------------------------
// Presets — quick configurations for common use cases
// ---------------------------------------------------------------------------

struct Preset {
    const char*  name;
    const char*  description;
    LaunchConfig cfg;
};

static const Preset PRESETS[] = {
    {
        "Smoke Test",
        "64 vectors, d=128, b=4 — quick sanity check after build",
        {64, "full"},
    },
    {
        "Medium Batch",
        "256 vectors — moderate validation run",
        {256, "full"},
    },
    {
        "Large Batch",
        "1024 vectors — stress test for tile pipelining",
        {1024, "full"},
    },
};

static constexpr int NUM_PRESETS = static_cast<int>(sizeof(PRESETS) / sizeof(PRESETS[0]));

// ---------------------------------------------------------------------------
// Terminal helpers
// ---------------------------------------------------------------------------

static void print_separator(char c = '-', int width = 60) {
    std::cout << std::string(width, c) << "\n";
}

static void print_header() {
    print_separator('=');
    std::cout << " Tenstorrent Wormhole — TurboQuant KV-Cache Launcher\n";
    print_separator('=');
    std::cout << "\n";
}

static std::string read_line(const std::string& fallback = "") {
    std::string line;
    std::getline(std::cin, line);
    if (!std::cin) { std::cout << "\n"; return fallback; }
    auto first = line.find_first_not_of(" \t\r\n");
    auto last  = line.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) return fallback;
    return line.substr(first, last - first + 1);
}

static uint32_t prompt_uint(const std::string& q, uint32_t def) {
    while (true) {
        std::cout << q << " [default: " << def << "]: ";
        std::string raw = read_line();
        if (raw.empty()) return def;
        try {
            int v = std::stoi(raw);
            if (v >= 1) return static_cast<uint32_t>(v);
        } catch (...) {}
        std::cout << "  Please enter a positive integer.\n";
    }
}

static bool prompt_yesno(const std::string& q, bool def = true) {
    std::cout << q << (def ? " [Y/n]: " : " [y/N]: ");
    std::string raw = read_line();
    if (raw.empty()) return def;
    return raw[0] == 'y' || raw[0] == 'Y';
}

// ---------------------------------------------------------------------------
// Preset menu + custom config
// ---------------------------------------------------------------------------

static void print_preset_menu() {
    std::cout << " Select an option:\n\n  [Presets]\n";
    for (int i = 0; i < NUM_PRESETS; ++i) {
        std::cout << "  " << (i + 1) << "  " << PRESETS[i].name << "\n"
                  << "     " << PRESETS[i].description << "\n";
    }
    std::cout << "\n  [Other]\n"
              << "  " << (NUM_PRESETS + 1) << "  Custom configuration\n"
              << "  0  Exit\n\n> ";
}

static LaunchConfig select_config() {
    while (true) {
        print_preset_menu();
        std::string raw = read_line();
        if (raw.empty()) continue;

        int choice = -1;
        try { choice = std::stoi(raw); } catch (...) {}

        if (choice == 0) { std::cout << "Goodbye.\n"; std::exit(0); }
        if (choice > 0 && choice <= NUM_PRESETS) return PRESETS[choice - 1].cfg;
        if (choice == NUM_PRESETS + 1) {
            LaunchConfig cfg;
            cfg.num_vectors = prompt_uint("Number of vectors to quantize", 64);
            return cfg;
        }
        std::cout << "  Invalid — enter 0–" << (NUM_PRESETS + 1) << ".\n\n";
    }
}

static void print_config(const LaunchConfig& cfg) {
    print_separator();
    std::cout << " Configuration summary:\n\n"
              << "  Vectors : " << cfg.num_vectors << "\n"
              << "  Mode    : " << cfg.mode << "\n";
    print_separator();
}

static std::vector<std::string> build_args(const LaunchConfig& cfg) {
    return {
        "--num-vectors", std::to_string(cfg.num_vectors),
        "--mode",        cfg.mode,
    };
}

static int run_host(const std::vector<std::string>& args) {
    std::vector<const char*> argv_ptrs;
    argv_ptrs.push_back(HOST_EXE);
    for (const auto& a : args) argv_ptrs.push_back(a.c_str());
    argv_ptrs.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) { std::cerr << "Error: fork() failed\n"; return -1; }
    if (pid == 0) {
        execvp(HOST_EXE, const_cast<char* const*>(argv_ptrs.data()));
        std::cerr << "Error: could not execute '" << HOST_EXE
                  << "' — is it compiled?\n";
        std::exit(1);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    print_header();
    LaunchConfig cfg = select_config();
    print_config(cfg);
    std::cout << "\n";
    if (!prompt_yesno("Run turboquant_host with the above settings?", true)) {
        std::cout << "Cancelled.\n";
        return 0;
    }
    std::cout << "\n";
    print_separator('=');
    int rc = run_host(build_args(cfg));
    print_separator('=');
    std::cout << (rc == 0 ? "turboquant_host finished successfully.\n"
                          : "turboquant_host exited with code " + std::to_string(rc) + ".\n");
    return rc;
}
