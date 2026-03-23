import { write } from "bun";
import { generateCstDts } from "chevrotain";
import { DataflowParser } from "../../parser";

const parser = new DataflowParser();
const productions = parser.getGAstProductions();
const dtsString = generateCstDts(productions);

await write("packages/compiler/src/types/cst-generated-types.d.ts", dtsString);