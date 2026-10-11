import assert from "node:assert/strict";
import { test } from "node:test";
import { closure, dynamicImports, staticImports } from "../scripts/bundle-size.mjs";

test("static and dynamic imports are told apart in minified output", () => {
  const code =
    'import{a as e}from"./a-1.js";import"./b-2.js";export{x}from"./c-3.js";const L=()=>import(`./Route-9.js`).then(m=>m);';
  assert.deepEqual(staticImports(code).sort(), ["a-1.js", "b-2.js", "c-3.js"]);
  assert.deepEqual(dynamicImports(code), ["Route-9.js"]);
});

test("the closure follows static edges only and survives cycles", () => {
  const files = {
    "e.js": 'import"./a.js";x=()=>import("./lazy.js")',
    "a.js": 'import{b}from"./b.js";',
    "b.js": 'import{a}from"./a.js";',
    "lazy.js": "",
  };
  assert.deepEqual([...closure(files, "e.js")].sort(), ["a.js", "b.js", "e.js"]);
});
