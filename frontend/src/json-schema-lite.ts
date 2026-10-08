/**
 * A small JSON Schema checker for the Playground: enough to say whether a tool call's arguments
 * or a structured reply follow the schema the user wrote (type, enum, const, required, properties,
 * additionalProperties, items, bounds, pattern, anyOf/oneOf). Keywords it does not know are
 * ignored, never treated as a pass for something it cannot judge: `unsupported` lists them.
 */
export interface SchemaIssue {
  path: string;
  message: string;
}

type Schema = Record<string, unknown>;
const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const KNOWN = new Set([
  "type",
  "enum",
  "const",
  "required",
  "properties",
  "additionalProperties",
  "items",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "minLength",
  "maxLength",
  "pattern",
  "minItems",
  "maxItems",
  "anyOf",
  "oneOf",
  // annotations that never change validity
  "description",
  "title",
  "default",
  "examples",
  "$schema",
  "$id",
  "strict",
]);

const typeOf = (v: unknown): string =>
  v === null
    ? "null"
    : Array.isArray(v)
      ? "array"
      : typeof v === "number"
        ? Number.isInteger(v)
          ? "integer"
          : "number"
        : typeof v;

const matchesType = (t: string, v: unknown) =>
  t === "number" ? typeof v === "number" : typeOf(v) === t;

/** Keywords in the schema that this checker does not evaluate. */
export function unsupportedKeywords(
  schema: unknown,
  found = new Set<string>(),
) {
  if (Array.isArray(schema))
    schema.forEach((s) => unsupportedKeywords(s, found));
  else if (isObj(schema)) {
    for (const [k, v] of Object.entries(schema)) {
      if (k === "properties" && isObj(v))
        Object.values(v).forEach((s) => unsupportedKeywords(s, found));
      else if (!KNOWN.has(k)) found.add(k);
      else if (
        k === "items" ||
        k === "anyOf" ||
        k === "oneOf" ||
        k === "additionalProperties"
      )
        unsupportedKeywords(v, found);
    }
  }
  return [...found].sort();
}

export function validateSchema(
  schema: unknown,
  value: unknown,
  path = "$",
): SchemaIssue[] {
  if (schema === true || !isObj(schema)) return [];
  const s = schema as Schema;
  const out: SchemaIssue[] = [];
  const add = (message: string) => out.push({ path, message });
  const types = s.type === undefined ? [] : ([] as unknown[]).concat(s.type);
  if (
    types.length &&
    !types.some((t) => typeof t === "string" && matchesType(t, value))
  )
    add(`type: expected ${types.join(" | ")}, got ${typeOf(value)}`);
  if (
    Array.isArray(s.enum) &&
    !s.enum.some((e) => JSON.stringify(e) === JSON.stringify(value))
  )
    add(
      `enum: ${JSON.stringify(value)} is not one of ${JSON.stringify(s.enum)}`,
    );
  if ("const" in s && JSON.stringify(s.const) !== JSON.stringify(value))
    add(`const: expected ${JSON.stringify(s.const)}`);
  if (typeof value === "number") {
    if (typeof s.minimum === "number" && value < s.minimum)
      add(`minimum: ${value} < ${s.minimum}`);
    if (typeof s.maximum === "number" && value > s.maximum)
      add(`maximum: ${value} > ${s.maximum}`);
    if (typeof s.exclusiveMinimum === "number" && value <= s.exclusiveMinimum)
      add(`exclusiveMinimum: ${value} <= ${s.exclusiveMinimum}`);
    if (typeof s.exclusiveMaximum === "number" && value >= s.exclusiveMaximum)
      add(`exclusiveMaximum: ${value} >= ${s.exclusiveMaximum}`);
  }
  if (typeof value === "string") {
    if (typeof s.minLength === "number" && value.length < s.minLength)
      add(`minLength: ${value.length} < ${s.minLength}`);
    if (typeof s.maxLength === "number" && value.length > s.maxLength)
      add(`maxLength: ${value.length} > ${s.maxLength}`);
    if (typeof s.pattern === "string") {
      try {
        if (!new RegExp(s.pattern).test(value))
          add(`pattern: does not match ${s.pattern}`);
      } catch {
        add(`pattern: ${s.pattern} is not a valid regular expression`);
      }
    }
  }
  if (Array.isArray(value)) {
    if (typeof s.minItems === "number" && value.length < s.minItems)
      add(`minItems: ${value.length} < ${s.minItems}`);
    if (typeof s.maxItems === "number" && value.length > s.maxItems)
      add(`maxItems: ${value.length} > ${s.maxItems}`);
    if (isObj(s.items))
      value.forEach((item, i) =>
        out.push(...validateSchema(s.items, item, `${path}[${i}]`)),
      );
  }
  if (isObj(value)) {
    if (Array.isArray(s.required))
      for (const k of s.required)
        if (typeof k === "string" && !(k in value))
          out.push({ path: `${path}.${k}`, message: "required: missing" });
    const props = isObj(s.properties) ? s.properties : {};
    for (const [k, sub] of Object.entries(props))
      if (k in value)
        out.push(...validateSchema(sub, value[k], `${path}.${k}`));
    if (s.additionalProperties === false)
      for (const k of Object.keys(value))
        if (!(k in props))
          out.push({
            path: `${path}.${k}`,
            message: "additionalProperties: not allowed",
          });
    if (isObj(s.additionalProperties))
      for (const k of Object.keys(value))
        if (!(k in props))
          out.push(
            ...validateSchema(s.additionalProperties, value[k], `${path}.${k}`),
          );
  }
  if (
    Array.isArray(s.anyOf) &&
    !s.anyOf.some((sub) => validateSchema(sub, value, path).length === 0)
  )
    add("anyOf: matches none of the alternatives");
  if (Array.isArray(s.oneOf)) {
    const n = s.oneOf.filter(
      (sub) => validateSchema(sub, value, path).length === 0,
    ).length;
    if (n !== 1) add(`oneOf: matches ${n} alternatives, expected exactly 1`);
  }
  return out;
}

/** Whether `text` is JSON that is itself a usable schema (an object). Error text is the parser's. */
export function parseSchemaText(
  text: string,
): { schema: Record<string, unknown> } | { error: string } {
  try {
    const v: unknown = JSON.parse(text);
    return isObj(v) ? { schema: v } : { error: "not-object" };
  } catch (e) {
    return { error: e instanceof Error ? e.message : "parse" };
  }
}
