/** Turn fenced diagrams into the same lazy Mermaid component as authored JSX.
 * Use a literal MDX attribute so quotes/braces in diagrams never become code.
 */
export function remarkMermaid() {
  return (tree) => {
    const walk = (parent) => {
      if (!Array.isArray(parent.children)) return;
      parent.children = parent.children.map((node) => {
        if (node.type === "code" && node.lang === "mermaid")
          return {
            type: "mdxJsxFlowElement",
            name: "Mermaid",
            attributes: [
              { type: "mdxJsxAttribute", name: "chart", value: node.value },
            ],
            children: [],
            position: node.position,
          };
        walk(node);
        return node;
      });
    };
    walk(tree);
  };
}
