"""Localizer (C1): the instrument. Find WHERE in the optimizer state the poison lives.

Given a (w, m, v) snapshot at a spike, score parameters/groups by how implicated they are,
via SNR of the Adam update, local curvature, and a combined poison score.
"""
