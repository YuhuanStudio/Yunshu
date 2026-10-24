import assert from "node:assert/strict";
import test from "node:test";
import { remarkMermaid } from "../scripts/remark-mermaid.mjs";

test("fenced Mermaid becomes a literal lazy component; other code metadata stays intact", () => {
  const chart = 'graph TD; A["quoted {text}"] --> B';
  const code = {
    type: "code",
    lang: "python",
    meta: 'title="example.py" {2}',
    value: "print(1)",
  };
  const tree = {
    type: "root",
    children: [
      {
        type: "blockquote",
        children: [{ type: "code", lang: "mermaid", value: chart }, code],
      },
    ],
  };
  remarkMermaid()(tree);
  const [diagram, other] = tree.children[0].children;
  assert.equal(diagram.type, "mdxJsxFlowElement");
  assert.equal(diagram.name, "Mermaid");
  assert.deepEqual(diagram.attributes, [
    { type: "mdxJsxAttribute", name: "chart", value: chart },
  ]);
  assert.equal(other, code);
});
