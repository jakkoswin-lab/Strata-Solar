// include/strata/glm4/engine.hpp - the Solar-Open-100B (GGUF architecture glm4moe) engine path as a library:
// `strata` hands a glm4moe model to it at the top of main() (the family switch), and strata-glm4 is a thin wrapper
// around the same entry point.  See src/glm4/glm4_engine.cu.
#pragma once

namespace strata::glm4 {

/// The engine's command line (strata's own flag names where they mean the same thing: --native / --model,
/// --max-context, --prefill / --chunk, --serve, --tokens, --max-new, --batch).  Returns the process exit code.
int run(int argc, char** argv);

}  // namespace strata::glm4
