# ADR 0011: UI Design Decisions — Editorial Cinematic System and Accessibility

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Section 7 defines an exact design specification titled **"Editorial Cinematic", Minimalist Editorial**. It specifies typography, letter-spacing, animations, color palette, responsive breakpoints, and hero background video styling.

## Decision
1. **Typography**:
   - Google Fonts: `Instrument Serif` (Display, 400 weight) and `Inter` (Body: 400 regular, 500 medium).
   - H1 Heading: `font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 80px (mobile 48px); line-height: 0.95; letter-spacing: -2.46px;`.
   - `<em>` elements within headings render with `font-style: normal;` for emphasis styling without italicization.
   - Sub-header: 18px; UI controls and buttons: 14px Inter Medium (500).
2. **Color Palette & Contrast**:
   - Primary text: `#0f172a` (Slate 900).
   - Muted text: `hsl(215, 25%, 32%)`.
   - Backgrounds: Hero video fallback `hsl(201, 100%, 13%)`; light sections `#ffffff` with thin border `hsl(215, 25%, 90%)`.
   - Action buttons: Pill buttons with `#000000` background and `#ffffff` text, `border-radius: 9999px`.
3. **Hero Video & Overlay**:
   - Background video: `https://designerstephen.github.io/public-assets/videos/serene-art-hero.mp4` with autoplay, loop, muted, playsinline.
   - Overlay: To ensure perfect legibility of typography while maintaining the cinematic video backdrop, an overlay of `rgba(255, 255, 255, 0.72)` or calibrated dark contrast overlay with crisp text color is applied. Text contrast meets WCAG AA standards (>4.5:1 ratio).
4. **Navigation & Labels**:
   - 3-column CSS Grid container (max-width 1280px, padding 24px 32px).
   - Right CTA: Pill button styled with label "Open the map" (replacing template text "Find my dream" to accurately match hazard prediction product).
5. **Entrance Animation**:
   - Keyframe `@keyframes fade-rise`: `from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); }` (0.8s ease-out).
   - Stagger: H1 at 0s, paragraph at 0.2s, button at 0.4s (`animation-fill-mode: both`). Hover scale `1.03` with 0.3s transition.
   - Full support for `@media (prefers-reduced-motion: reduce)`.

## Consequences
- Produces a world-class, editorial aesthetic that impresses users immediately while adhering strictly to typographic and accessibility standards.
