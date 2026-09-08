# 1. Direct Max Similarity Matching for Target Identification

**Date**: 2026-08-27  
**Status**: Accepted

## Context & Decision
Previously, when a target image contained multiple faces or when multiple reference photos were uploaded, the matching engine computed an arithmetic centroid vector of all target embeddings and calculated similarity using a blended formula:
$$\text{Sim} = 0.65 \times \text{MaxSim} + 0.35 \times \text{CentroidSim}$$

When a user uploaded a group photo containing multiple different individuals (e.g. 11 people), the centroid represented the average vector of unrelated humans, having a very low cosine similarity (~0.18) to any candidate. This diluted genuine matches of 62.6% down to 47.0%, dropping them below the threshold.

We decided to eliminate centroid dilution entirely and adopt **Direct Max Similarity**:
$$\text{Sim} = \max_{t \in \text{TargetEmbeddings}} (t \cdot c)$$

Additionally, we introduced an interactive face selection workflow (`/api/v1/target/detect-faces`) allowing the user to select the specific person if an uploaded image contains multiple faces, defaulting to the primary face (Face #0) rather than averaging different individuals.

## Consequences
- **Positive**: Accurate recognition scores (e.g. 62.6% stays 62.6%), completely eliminating false rejections caused by group photos.
- **Positive**: Supports multi-view targets (e.g. straight on, angled, wearing glasses) where matching against any valid angle yields a true positive.
- **Trade-off**: Requires $O(N \times M)$ dot products between $N$ target views and $M$ candidates, which is trivial ($<0.1$ms for $N \le 10$) on modern hardware.
