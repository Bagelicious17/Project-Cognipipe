---
name: CodeNotebook
colors:
  primary: "#292524"
  secondary: "#78716C"
  success: "#10B981"
  warning: "#F59E0B"
  danger: "#EF4444"
  info: "#3B82F6"
  background: "#F3F2ED"
  surface: "#FFFFFF"
  foreground: "#292524"
  border: "#E7E5E4"
  overlay: "rgba(0, 0, 0, 0.5)"
colors-dark:
  primary: "#a8a29e"
  secondary: "#a8a29e"
  success: "#34D399"
  warning: "#FBBF24"
  danger: "#F87171"
  info: "#60A5FA"
  background: "#1C1917"
  surface: "#292524"
  foreground: "#F3F2ED"
  border: "#44403C"
typography:
  h1:
    fontFamily: "Inter, sans-serif"
    fontSize: 32px
    fontWeight: 600
    lineHeight: 1.2
  h2:
    fontFamily: "Inter, sans-serif"
    fontSize: 24px
    fontWeight: 600
    lineHeight: 1.3
  h3:
    fontFamily: "Inter, sans-serif"
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.35
  body-md:
    fontFamily: "Inter, sans-serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.5
  body-sm:
    fontFamily: "Inter, sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.45
  body-xs:
    fontFamily: "Inter, sans-serif"
    fontSize: 12px
    fontWeight: 500
    lineHeight: 1.33
spacing:
  xs: 0.25rem
  sm: 0.5rem
  md: 1rem
  lg: 1.5rem
  xl: 2rem
rounded:
  sm: 0.125rem
  md: 0.5rem
  lg: 0.75rem
  xl: 1rem
  full: 9999px
---

## Overview

CodeNotebook embodies the calm clarity of a well-worn notebook — warm neutrals, subtle stone textures, and serif typography that invites deep focus. The design language is professional yet approachable — a space where code thoughts flow naturally onto the page.

## Colors

The palette is anchored by #292524 as the primary accent, with #a8a29e for dark mode.

### Foundation

CodeNotebook's color story centers on **warm stone** (#292524) — a deep, friendly charcoal that conveys professionalism without coldness. This is paired with cream-toned backgrounds and subtle warm accents.

### The Warm Neutral System

**Primary — Stone (#292524)**: The signature dark — used for primary actions, headings, and key text. Not pure black; warm and approachable.

**Secondary — Warm Gray (#78716C)**: A muted stone for secondary elements. Provides hierarchy without harsh contrast.

### Surface Hierarchy

| Level | Light Mode | Dark Mode | Use |
|-------|-----------|-----------|-----|
| Canvas | #F3F2ED | #1C1917 | Warm cream background |
| Surface | #FFFFFF | #292524 | Cards, elevated elements |
| Muted | #F5F5F4 | #292524 | Subtle backgrounds |
| Subtle | #FAFAF9 | #44403C | Dividers, separators |

### Dark Mode: "Nighttime Focus"

Dark mode transforms CodeNotebook into a comfortable nighttime workspace — deep charcoal surfaces maintain the warm stone feel while reducing eye strain for extended coding sessions.

**Core Principles:**

1. **Stone Inverts**: The warm stone (#292524) becomes lighter in dark mode (#a8a29e), maintaining warmth
2. **Cream Becomes Charcoal**: The canvas shifts from cream to deep charcoal (#1C1917)
3. **Warmth Preserved**: Even in dark mode, the palette maintains its approachable, non-cold feel

**Dark Mode Token Mapping:**

| Light Mode | Dark Mode | Rationale |
|------------|-----------|-----------|
| #292524 (Primary) | #a8a29e | Stone lightens for dark contrast |
| #F3F2ED (Canvas) | #1C1917 | Cream becomes charcoal |
| #FFFFFF (Surface) | #292524 | Cards adapt to dark context |
| #78716C (Secondary) | #a8a29e | Gray adapts for hierarchy |

### Semantic Colors

| Token | Light | Dark |
|-------|-------|------|
| Background | #F3F2ED | #1C1917 |
| Surface | #FFFFFF | #292524 |
| Foreground | #292524 | #F3F2ED |
| Border | #E7E5E4 | #44403C |
| Primary | #292524 | #a8a29e |
| Secondary | #78716C | #a8a29e |
| Success | #10B981 | #34D399 |
| Warning | #F59E0B | #FBBF24 |
| Danger | #EF4444 | #F87171 |
| Info | #3B82F6 | #60A5FA |

### Signature Details

- **Warm Canvas**: #F3F2ED creates the "paper" feel — not stark white, not boring gray, but warm cream that invites writing
- **Stone Primary**: #292524 is friendlier than pure black — readable but not harsh
- **Subtle Elevation**: Shadows are soft and minimal — the warmth comes from color, not dramatic depth

## Typography

### Font Stack

**Inter** — Clean, modern sans-serif for interface clarity. Paired with **Playfair Display** serif for headings that invite deep reading. The serif gives CodeNotebook its notebook-like character.

### Type Scale

| Role | Size | Weight | Line Height | Character |
|------|------|--------|-------------|-----------|
| H1 | 32px | 600 | 1.2 | Page titles |
| H2 | 24px | 600 | 1.3 | Section headers |
| H3 | 20px | 600 | 1.35 | Card titles |
| Body | 16px | 400 | 1.5 | Content, code |
| Small | 14px | 400 | 1.45 | Secondary text |
| Micro | 12px | 500 | 1.33 | Labels, badges |

### The Warm Text Rule

Text colors use warm undertones — not pure gray but stone gray (#57534E). This subtle warmth makes long reading sessions more comfortable.

## Layout & Spacing

The spacing system follows a 4px grid scale based on 0.25rem increments:

| Token | Value | Usage |
|-------|-------|-------|
| xs | 0.25rem | Micro spacing |
| sm | 0.5rem | Element gaps |
| md | 1rem | Standard padding |
| lg | 1.5rem | Section gaps |
| xl | 2rem | Page margins |

## Elevation & Depth

### Shadow Philosophy

CodeNotebook uses **minimal, functional shadows** — subtle depth that doesn't distract from content. The focus is on color contrast, not dramatic elevation.

| Level | Treatment | Use |
|-------|-----------|-----|
| Level 0 | none | Flat elements |
| Level 1 | 0 1px 3px rgba(0,0,0,0.1) | Subtle hover states |
| Level 2 | 0 4px 6px rgba(0,0,0,0.1) | Cards, dropdowns |
| Level 3 | 0 10px 25px rgba(0,0,0,0.15) | Modals, elevated panels |

### Border Usage

Very subtle borders (1px, #E7E5E4) for definition. CodeNotebook relies on color contrast more than borders for hierarchy.

## Shapes

The shape language uses 0.125rem as the base corner radius, keeping elements soft and approachable:

| Token | Value | Usage |
|-------|-------|-------|
| sm | 0.125rem | Subtle rounding |
| md | 0.5rem | Buttons, inputs |
| lg | 0.75rem | Cards |
| xl | 1rem | Modals, containers |
| full | 9999px | Pills, badges |

The minimal 0.125rem sm radius creates subtle softening at the edges. Buttons and inputs use 0.5rem md radius — rounded enough to feel intentional.

## Components

### Buttons & Interaction

**Primary CTA**: Stone (#292524) background, white text, minimal shadow. On hover: darkens slightly. On press: further darkens. The understated, professional action.

**Secondary**: Warm gray (#78716C) background for secondary actions. Establishes hierarchy without competition.

**Ghost Buttons**: Transparent with subtle stone border — for tertiary actions.

### Cards & Containers

**Cards**: White surface, very subtle border, minimal shadow. The card feels like a clean page in the notebook.

**Code Blocks**: Slightly darker background (#F5F5F4), subtle border, monospace font — the "written code" on the notebook page.

### Inputs & Selection

**Text Inputs**: White background, subtle stone border (#E7E5E4), 4px radius. Focus state: primary border. Placeholder uses warm gray (#A8A29E).

**Select Dropdowns**: Consistent with input styling, chevron indicator. Dropdown panel uses level-2 shadow.

**Checkboxes**: Custom styled — stone check when active.

**Switches**: iOS-style toggle — stone when active, gray when inactive.

### Feedback Components

**Alerts**: Left-bordered (4px) in semantic color, white background, subtle border.

**Toasts**: Floating panel with soft shadow, icon + message.

**Badges**: Small pill shape, solid color or subtle tint.

## Do's and Don'ts

### Do

- **Use warm stone for primary text** — #292524 is readable but not harsh
- **Apply the warm cream canvas** — #F3F2ED creates the notebook paper feel
- **Keep shadows minimal** — subtle depth, not dramatic elevation
- **Use warm gray for secondary text** — #78716C not pure gray
- **Maintain warmth in dark mode** — stone lightens, doesn't become cold blue-gray
- **Use subtle borders** — 1px borders for definition, not heavy outlines

### Don't

- **Don't use pure black text** — warm stone (#292524) is more approachable
- **Don't use aggressive shadows** — keep elevation soft, like paper layers
- **Don't use cold grays** — warm undertones throughout
- **Don't use stark white canvas** — cream (#F3F2ED) creates the notebook feel
- **Don't use heavy borders** — subtle 1px definition only
- **Don't make elements feel floating** — cards should feel like pages, not floating panels
