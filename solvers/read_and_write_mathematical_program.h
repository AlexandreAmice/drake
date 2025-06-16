#include "drake/solvers/mathematical_program.h"
namespace drake {
namespace solvers {

/**
 * Write a convex MathematicalProgram to a file in CBF format.
 *
 * The Conic Benchmark Format (CBF) is a file format for representing conic
 * programs which is easy to read and write. It is defined in Mosek's
 * documentation: https://docs.mosek.com/latest/capi/cbf-format.html
 *
 * @param prog: The program to serialize.
 * @param filename: The file to write to. If the file already exists, it will be
 * overwritten.
 * @throws std::runtime_error if the program is not convex.
 */
void WriteMathematicalProgramToCbfFormat(const MathematicalProgram& prog,
                                         const std::string& filename);

/**
 * Reads a CBF file into a MathematicalProgram.
 * @param filename: The file to read from.
 */
MathematicalProgram ReadMathematicalProgramFromCbfFormat(
    const std::string& filename);

}  // namespace solvers
}  // namespace drake