---
name: yunui
description: Build or update a React app that the user explicitly wants based on YunUI, using installed YunUI components and compositions. Use for YunUI integration, component selection and setup failures; not for another UI kit or for merely comparing visual references.
---

# Use YunUI as the UI implementation

YunUI is a versioned component dependency. Use its exported React components
for controls and page patterns. Looking at screenshots or importing only its
colors does not complete a request to build with YunUI.

## Resolve the installed system

Read `.yunui/context.json` when present. Use the installed `yunui` CLI through
the project's package manager (for example `pnpm exec yunui`). Run:

```sh
yunui doctor --strict
yunui find dialog
yunui info ./chat/ChatComposer
```

Fix dependency, stylesheet and Tailwind scanning failures before inventing
replacement styling. The installed declarations and catalogue are the API
authority; online examples may describe a different revision. If the project
uses an npm alias, use its `consumerImport`, not a nonexistent scoped module.
Geist and JetBrains Mono are host-loaded fonts; do not compensate for a missing
font with a different component implementation.

## Select, then compose

Map every UI surface you intend to introduce to exports before writing it,
including optional extras such as a profile avatar or an empty state. A short mapping
in the work update or a component plan is sufficient; record a real feature gap
instead of treating unfamiliar props as a missing component.

| Surface | Start with |
| --- | --- |
| Basic controls / data | core Button, Input, Card, Dialog, Select, Table |
| Identity / feedback | core Avatar + AvatarFallback, Badge, EmptyState, Alert, Skeleton / Spinner |
| Landing / product pages | patterns Navbar, PageLayout, MarketingHero, FeatureCard, CTASection |
| Dashboard / settings | patterns PageHeader, StatCard, Sidebar, SettingsShell, SettingRow |
| Conversation | chat ChatHeader, ChatMessageList, ChatMessage, ChatComposer |
| Answer sources / follow-ups / retrieval | chat StreamingText, ContextCards |
| Selected passage edits / proposed changes | ai SelectionActions, DiffTable |
| Agent process | ai AgentTimeline, AgentRunStatus; content MarkdownRenderer / InlineCitation |

`CodeBlock` exists in both `/patterns` and `/content`; query the entry-specific
API. `Dialog` is the default accessible dialog; do not rebuild it with a portal.
Use the packaged recipes identified by `.yunui/context.json` for complete,
typechecked import compositions.

Import existing exports or an existing app facade that re-exports them. Keep
product data, routes, request handlers and application state in the host. Native
structural markup and layout utilities are fine. Do not copy YunUI component
source, create a second Button/Card/Dialog, replace its controls with another
kit, or approximate the house design with hand-written control CSS.

When behavior is missing, first use the documented props, slots and composition.
A small product-specific wrapper is appropriate when it delegates UI to YunUI.
Identify any remaining library gap and propose an upstream change within the
user's authorized scope. Respect repository design-review requirements; an
external reference is not approval to change the design system.

## Verify actual reuse

Run the host's typecheck/build, `yunui doctor --strict` and `yunui audit --strict`.
Read every audit candidate: replace a recreated avatar/empty state with its
export, or explain the product-specific need. A brand mark and record text are
host content; a generic profile avatar is a reusable UI surface. Inspect actual
imports and wrapper origins; a doctor result checks integration, not whether
every screen correctly reused components. Render the requested app and inspect
its UI at the relevant sizes/themes. Report which surfaces use which exports
and any justified product-specific UI. Do not call a page YunUI-based because
it only resembles the showcase.

For compiler integration, see the packaged `agent/recipes/setup.md` beside the
typed recipes. A stylesheet containing Tailwind directives must be compiled by
the host; serving that source CSS directly is not a working setup.


For answer sources, follow-ups, retrieval context, passage edits and proposed change review, use the public units in `recipes/conversation-units.tsx`. Do not rebuild their toolbar, source list, selection marks or apply footer in the application. Host props own stream progress, selection and backend results; callbacks request work and do not prove execution succeeded.
