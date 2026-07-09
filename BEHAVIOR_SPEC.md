# Behavior Spec

> The gate. One or two sentences a stranger could use to mark any output pass/fail.
> This is simultaneously the data-generation rubric, the eval criterion, and the spiky POV.

## Spec

> Given a single Japanese sentence, the model returns an English translation whose
> politeness register — **casual / neutral-polite / formal-deferential** — matches the
> register of the Japanese source as determined by its verb morphology. It never flattens
> a formal or humble source into casual English, nor inflates a casual source into formal
> English.

## Passes the litmus test?

**Yes.** A well-prompted small base model does *not* reliably hold register — it flattens
toward neutral-polite (its training prior), silently and consistently. That is a
reliability gap fine-tuning closes and prompting can't guarantee.

## Operational definition of the three English bands

The Japanese side is verb-encoded and machine-detectable. The **English** side is diffuse,
so it must be pinned down or the eval is noise. An output is scored against these markers:

| Band | English realized as |
|---|---|
| **casual** | contractions, phrasal verbs, "gonna/kinda", direct imperatives, "yeah/nope", slang |
| **neutral-polite** | full forms, "please / could you", standard register, no slang, no elevated diction |
| **formal-deferential** | "I would be grateful if…", nominalizations, hedged/indirect requests, honorific address, no contractions |

## The forbidden failure (what the behavioral check hunts for)

**Flattening**: a formal/humble source rendered as neutral or casual English (or, less
often, a casual source inflated to formal). The single most important reported number is
the **flattening rate**.

## Register-band mapping (JP → target English band)

| Japanese (keigo) | Band |
|---|---|
| plain / jōtai (常体) | casual |
| teineigo (丁寧語, です/ます) | neutral-polite |
| sonkeigo (尊敬語) / kenjōgo (謙譲語) | formal-deferential |
