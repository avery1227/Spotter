"""Things in the sky that are not in a track feed: satellites and lightning.

Both are positioned in 3D like aircraft and ships and projected through the
same calibrated camera, but neither fits the report/interpolate model of the
track store: a satellite's position at any instant is computed from its
orbit, and a lightning strike is a single event. Each layer therefore turns
a frame time straight into :class:`~spotter.projection.ProjectedTarget`
objects, which the renderer draws alongside everything else.
"""
