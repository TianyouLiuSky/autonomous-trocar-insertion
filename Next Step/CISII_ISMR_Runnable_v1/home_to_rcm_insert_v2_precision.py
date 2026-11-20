#!/usr/bin/env python3
"""PROPOSED PRECISION UPGRADE of home_to_rcm_insert.py -- NOT yet promoted to
the live script. Written for review by a human before running on the real
robot: it changes the ORIENT control law and adds new logging/diagnostics on
top of the validated 2026-08-13 baseline (home_to_rcm_insert.py, same dir).

WHY THIS FILE EXISTS INSTEAD OF EDITING home_to_rcm_insert.py IN PLACE:
this script commands real actuator velocities on a physical eye-surgery
robot (SHER 2.0). Changes to the control law should be diffed and read by a
human, then run manually in a controlled setting, before they touch the
script actually used for a live demo. Diff this against home_to_rcm_insert.py
to see exactly what changed; once you've bench-tested it, you can rename it
over the original (or just point your launch command at this file).

WHAT CHANGED vs the 2026-08-13 baseline (see CHANGELOG docstring at bottom of
each changed function for the specific reasoning):

1. auto_orient_about_rcm(): control law replaced. OLD: constant ASPEED with a
   simple ease-in over the last 5deg and NO floor -- observed in the
   2026-08-13 log (Matthew/logs/home_to_rcm_gui_final.log) to stall a few
   degrees short and time out on the first 't' press:
       "ORIENT(30deg) TIMEOUT (4.7deg remaining)"
       "ORIENT(30deg) reached (1.00deg)"   <- only succeeded on retry
   NEW: proportional band (ORIENT_EASE_DEG) + integral term for steady
   friction/backlash bias (ORIENT_KI, anti-windup clamped) + a minimum-speed
   floor (ORIENT_MIN_OMEGA) so it can never fully stall while err>tol. Default
   tolerance tightened 1.0deg -> 0.3deg now that the controller can actually
   reach it without stalling.

2. desired_insertion_dir(): added an OPTIONAL deterministic azimuth
   (INSERT_AZIMUTH_DEG). The baseline picks the azimuth of the 30deg tilt
   from "whatever direction the tool happened to be approaching from" --
   fine for a single demo, but not reproducible run-to-run or across
   approach trajectories. If INSERT_AZIMUTH_DEG is unset, behavior is
   UNCHANGED (100% backward compatible).

3. NEW: precision_log.csv -- every completed ORIENT/APPROACH/INSERT/RETRACT
   appends one row (timestamp, action, target, achieved error, elapsed
   time, result). Lets you run the sequence N times in one session and
   compute real mean/std/max precision instead of citing a single video.

4. REVISED (2026-08-18, v2): measure_shaft_direction_3d() -- an
   INDEPENDENT 3D measurement of the physical needle-shaft direction, using
   ONLY the depth camera already on this rig (single RGBD camera, no
   second/stereo camera -- the CIS II report already showed stereo through
   the Leica is geometrically infeasible at this working distance, so this
   deliberately does NOT depend on that). An earlier draft of this check
   used color pixels only (a 2D image-plane angle), which is fundamentally
   NOT a 3D measurement -- a 2D line's image-plane angle cannot distinguish
   different 3D directions that project to the same 2D angle (the
   out-of-plane/depth-direction component is invisible to it). That draft
   also silently could not have worked at all: this script's RealSense
   pipeline only opened the COLOR stream, not depth.

   The corrected method:
     a) detect 2D shaft line segments (Canny + HoughLinesP) in a ROI around
        the projected pivot point, filtered first by proximity to the
        EXPECTED 2D angle (projecting target_dir through hand-eye+
        intrinsics, as a detection prior) then merged with other segments
        close to the best match's own angle (self-consistency);
     b) sample points along those segments and look up MIN depth (not
        mean/median) in a small window at each -- same reasoning
        auto_target_vision.py already uses for the thin needle tip: a
        window average over a thin foreground object against background
        is biased toward the background, min is not;
     c) deproject each (pixel, depth) to a 3D point in camera frame, then
        to BASE frame via R_he/t_he (the SAME hand-eye transform used
        everywhere else in this file -- so this check is independent of
        NEEDLE_AXIS_LOCAL / pivot calibration / FrameEE orientation, which
        is exactly what needs independent checking, but it is NOT
        independent of hand-eye calibration, which everything in this file
        shares; that scope is stated explicitly, not oversold);
     d) fit a 3D LINE through the resulting point cloud by total-least-
        squares (SVD of the centered points; direction = eigenvector of
        the LARGEST singular value -- the dual of phantom_normal_estimate.
        py's plane fit, which takes the SMALLEST singular vector as the
        normal; here we want the direction of maximum variance, i.e. the
        direction the points are spread along);
     e) compare this measured direction to the kinematically-commanded
        target_dir via arccos(|dot(d_hat, target_hat)|) -- the absolute
        value removes the line's inherent sign ambiguity (d and -d fit
        the same points equally well) from ever leaking into the reported
        angle.
   CONDITIONING CHECKS (won't report a falsely-precise number from a
   badly-conditioned fit -- the same principle phantom_normal_estimate.py
   already applies when it rejects too-small patches): requires >=
   VIS3D_MIN_POINTS valid 3D points AND >= VIS3D_MIN_EXTENT_MM of spatial
   spread along the fitted direction (a short segment cannot constrain a
   3D direction -- this is the same "small patch of a large sphere" ill-
   conditioning already documented in phantom_normal_estimate.py, just for
   a line instead of a plane), and reports an approximate angular
   uncertainty (small-angle: atan(fit_rms_mm / (extent_mm/2))) so a caller
   can see when the measurement itself is too noisy to be a meaningful
   cross-check, instead of a bare pass/fail. Failure to meet these never
   blocks the workflow -- it just skips the advisory print/log.

5. NEW: real BI-PLANAR insertion (was a single-angle straight-line move).
   The JHU CIS II Group 26 final report ("Autonomous Scleral Trocar
   Insertion", 2026-05-04 -- the direct ancestor of this codebase; see
   ~/final_report_cis_ii_-_autonomous_scleral_trocar_insertion.pdf) defines
   the clinical "Oblique Insertion Technique" this whole project targets as
   TWO planes, not one:
     a) oblique tunnel: shaft at INSERT_ANGLE_DEG (e.g. 30deg) off the
        surface normal, pivoting about the SURFACE RCM, advancing
        TUNNEL_DEPTH_MM (1.5mm) -- creates the self-sealing scleral tunnel.
     b) internal RCM shift: the tip position at the end of (a) becomes a
        NEW pivot point (the report's "internal RCM").
     c) vertical plunge: the shaft re-orients from 30deg back to 0deg
        (perpendicular / straight along the normal) pivoting about THAT
        internal RCM (not the surface RCM), then advances PLUNGE_DEPTH_MM
        (3.0mm) straight in -- this is the actual vitreous puncture.
   The 2026-08-13 baseline's docstring already CITED these two numbers
   (1.5mm + 3.0mm = 4.5mm) but the actual 'i' key just did one straight-line
   move at the fixed 30deg angle for the FULL 4.5mm -- it never performed
   step (b)/(c), so the code's documented intent and its actual behavior
   had diverged. This version implements the real two-phase sequence:
     'i' = TUNNEL   (oblique, pivots about the surface RCM, arms 'p')
     'p' = PLUNGE    (re-orient to perpendicular about the NEW internal RCM,
                      then advance the vertical plunge; only armed right
                      after a successful TUNNEL)
   Set BIPLANAR_MODE=0 to fall back to the old single-angle straight-line
   'i' behavior (kept for A/B comparison, see precision_report.py).

6. NEW (2026-08-18): live, IN-PROCESS RCM ring detection + phantom-normal
   estimation -- ports the algorithms from auto_target_vision.py ('z' RCM
   lock) and phantom_normal_estimate.py ('n', new key) into THIS script's
   own already-open camera pipe, instead of requiring a SEPARATE terminal to
   hold the RealSense and write auto_target.npz for this script to read.
   That two-terminal handoff was fragile in practice: home_to_rcm_insert.py's
   AUTO_TARGET_MAX_AGE_S default (5.0s) requires 'z' to be pressed within a
   few seconds of the OTHER process's last write, and the two processes
   cannot both hold the camera at once, so the realistic workflow was
   "stop process A right after it locks, immediately start this script,
   press z within the staleness window" -- easy to miss, and 'z' then just
   silently does nothing with no useful diagnostic.
     'z' now runs the SAME ring-detection algorithm (CIELAB a* threshold +
     Kasa circle fit + shape-residual rejection, see detect_ring()) on every
     displayed frame, continuously accumulating the same rolling
     smoothing/jitter buffer auto_target_vision.py used, and reports LIVE
     status ("ring not visible: <reason>" / "collecting N/15" / "LOCKED")
     instead of a silent no-op -- 'z' just asks "is it locked yet", it no
     longer triggers a one-shot detection.
     'n' (new) runs the SAME phantom-normal local-plane-fit algorithm
     (bowl-only, red-rim color-excluded, see phantom_normal_estimate.py's
     METHOD HISTORY for why this exact method) once RCM is locked, using one
     fresh aligned frame from the SAME pipe -- no separate terminal, no
     camera handoff, no staleness window to race against.
   Both still write their npz artifacts (auto_target.npz / phantom_normal.npz)
   atomically on success, so the original standalone scripts remain valid as
   independent tools/artifacts if you ever want to run them separately (e.g.
   to sanity-check this integration against the original implementation).

7. NEW (2026-08-18): removed the MAX_TRAVEL_MM abort-mid-travel safety cap by
   default (it was tripping on normal HOME->hover/insert moves, not just
   runaway ones -- set MAX_TRAVEL_MM to a positive number via env var if you
   want that cap back), AND added a hard requirement that 'g'/APPROACH
   refuses to run without a FRESH, successful 't'/ORIENT on record
   (S.oriented, invalidated by any manual jog, re-homing, or re-locking RCM
   in between). Motivation: previously NOTHING enforced 't' before 'g' --
   pressing 'g' after a timed-out/skipped ORIENT (or after jogging away from
   a good one) silently inserted along whatever direction the tip happened
   to already be pointing, not the calibrated INSERT_ANGLE_DEG tilt. This
   was flagged after a demo where the achieved angle looked wrong and the
   tip did not visibly land on the marked RCM point -- this specific failure
   mode (operating on a stale/absent orientation) is now impossible; see the
   README's "补充5" for the other candidate root causes it does NOT rule out
   (tip-offset/pivot-cal bias, hand-eye bias, NEEDLE_AXIS_LOCAL bias -- none
   of which the software's own error readout can see, since they'd bias the
   software's internal model of "aligned"/"at the RCM" together with the
   real world).

8. NEW (2026-08-18): load_tip_offset_pivot() now also surfaces the fit_rms /
   loo_rms / orientation_spread_deg / age that pivot_calibration.py already
   computes and saves -- previously loaded and then silently discarded here.
   Printed at startup with a GOOD/MARGINAL/POOR/UNKNOWN verdict. This matters
   specifically because TIP_D from THIS SAME npz sets NEEDLE_AXIS_LOCAL a few
   lines below (when available) -- i.e. one calibration number simultaneously
   defines (a) where the software thinks the tip is and (b) what "aligned
   with target_dir" even means for the orientation controller. A biased pivot
   calibration would make BOTH the achieved angle and the RCM-insertion point
   wrong TOGETHER, in a way auto_orient_about_rcm()'s own err_deg<0.3deg
   convergence check is structurally unable to detect (it is only ever
   comparing the software's internal model against itself). This is now the
   single most actionable, no-new-hardware-needed first diagnostic step for
   "the achieved angle looks wrong / tip does not land on the RCM" -- if the
   printed verdict is MARGINAL/POOR, re-run pivot_calibration.py with a wider
   orientation sweep before trusting anything else about the demo's accuracy.

9. FIXED (2026-08-18): manual ANGULAR jog (k/m/j/l/u/o) published the
   angular velocity with a ZERO linear velocity -- i.e. it rotated about
   whatever pivot the robot's OWN low-level controller happens to default
   to, NOT about the RCM this script tracks. auto_orient_about_rcm() (the
   'automatic' 't'/PLUNGE-reorient path) always publishes
   v_lin = omega x lever to hold the tip on the RCM while rotating; the
   manual jog path never got the same treatment. This is exactly the
   "adjusting orientation near/at the RCM displaces the tip, not just
   rotates it" bug reported after the previous round of fixes: any fine
   manual orientation tweak (which is a very natural thing to do right
   at/near the RCM) had NO pivot compensation at all. Fixed by computing
   the same omega x lever compensation for manual ANG jog whenever RCM is
   latched, pivoting about S.internal_rcm once TUNNELed (matching 'p'
   REORIENT's own pivot choice) else the latched surface RCM; falls back to
   the old (uncompensated) behavior only if RCM isn't latched, since there
   is then no pivot reference to hold.

10. CHANGED (2026-08-18): keyboard sequence reordered from
    t(ORIENT) -> g(APPROACH) -> i(TUNNEL) to
    g(APPROACH) -> t(ORIENT) -> i(TUNNEL).
    Reasoning: with the OLD order, 't' ran while the tip was still at HOME
    -- far from the RCM -- so auto_orient_about_rcm() pivoted about the
    (distant) surface RCM with a large lever arm. Even though the ANGULAR
    error converged correctly, the required linear compensation
    (v_lin = omega x lever) swept the tip through a wide physical arc to
    get there ("绕了一大圈"), which is correct pivoting math but an
    unnecessarily large, visually confusing motion for a demo, and wastes
    the small workspace margin near the phantom. Swapping the order fixes
    this two ways at once:
      - compute_goals() now places hover_goal/insert_goal along the FIXED
        calibrated target direction (desired_insertion_dir()) instead of
        the current instantaneous tip->RCM line, so 'g' can safely run
        BEFORE orientation is corrected -- the hover point is already on
        the right axis regardless of which way the tip happens to be
        pointing at HOME.
      - 't' now pivots about the tip's OWN current position (captured once
        right when 't' is pressed), not the surface RCM. Since the lever
        arm starts at ~zero, the tip stays essentially fixed at the hover
        point while only the shaft direction rotates -- "reach the hover
        point, then rotate to the correct orientation without
        displacement," as requested.
    State-machine gating flipped to match: 'g' no longer requires a fresh
    't'; 't' now requires a fresh 'g'; 'i' requires both. Any manual jog,
    re-home ('h'), or RCM re-lock ('z') invalidates BOTH S.approached and
    S.oriented (previously only S.oriented), since either kind of motion
    can move the tip away from the hover point this scheme depends on.

11. FIXED (2026-08-18, real bug): the omega x lever pivot compensation
    (auto_orient_about_rcm() and the manual ANG-jog path added in pt.9) used
    `lever = tip - pivot`, i.e. it computed the lever arm from the TIP. But
    the published Vector3 linear command drives the EE ORIGIN's velocity,
    not the tip's, and true rigid-body kinematics gives
        v_tip = v_EE_origin + omega x (tip - EE_origin).
    Solving for the EE velocity that keeps the tip tracing a sphere around
    `pivot` (v_tip = omega x (tip - pivot), the "as if rigidly pivoting at
    pivot" model) gives
        v_EE_origin = omega x (EE_origin - pivot),
    i.e. the lever belongs between the EE ORIGIN and pivot, not between the
    tip and pivot -- using tip-pivot silently under/over-compensates by
    exactly the tip offset, and is only correct in the degenerate case of
    zero tool offset. This was most visible for the new self-pivot 't'
    (pt.10): pivot ~= tip at t=0, so lever ~= 0 throughout, meaning the code
    was commanding ~zero linear velocity while still commanding a real
    angular velocity -- the EE origin sat still while the tool rotated about
    it, sweeping the (offset) tip through a visible arc. This is exactly the
    "'t' still produces displacement" report. Fixed by computing the lever
    from `_ee["p"]` (EE origin) instead of `cur_tip()` at both call sites.

12. FIXED (2026-08-18): 't' (and 'g' via S.rcm_source=="vision") required
    the vision RCM lock to be less than AUTO_TARGET_MAX_AGE_S (5s) old,
    forcing 'z' to be re-pressed every few seconds even though nothing was
    wrong. That gate dates from the OLD two-process file-handoff design
    (pt.6) where a stale timestamp meant "the other process may have died";
    it stopped making sense once detection moved in-process (pt.6) -- the
    phantom is rigid and the locked point does not go stale just because
    time passed. Removed the staleness check entirely: once 'z' succeeds,
    the lock is valid indefinitely. Press 'z' again only if the
    phantom/camera actually physically moved.

13. CHANGED (2026-08-18): ORIENT_TIMEOUT_S (default 40s) now disabled by
    default (<=0 disables, same pattern as MAX_TRAVEL_MM in pt.7). Reported
    symptom: after a TIMEOUT aborted a slow-converging 't', pressing 't'
    again would drift further from the hover point each retry. Root cause
    is NOT the timeout check itself, but what a retry does: 't' captures a
    brand-new pivot_point = cur_tip() every time it's called (by design,
    pt.10 -- "rotate about wherever the tip is right now"). The
    omega x lever compensation (pt.11) holds the tip near its pivot only to
    first order per control tick; over a long, slow run (e.g. stuck easing
    near ORIENT_TOL_DEG) small real-world errors (actuator lag, encoder
    noise, residual hand-eye/pivot bias) can accumulate into a real, if
    small, net drift by the time it times out. A fresh retry then re-anchors
    pivot_point to THIS ALREADY-DRIFTED position instead of the original
    hover point -- drift is not self-correcting, it ratchets further away
    on every timeout+retry cycle. Removing the timeout means a single 't'
    press keeps running (using the SAME original pivot the whole time)
    until it actually converges, instead of restarting from an increasingly
    drifted position. The separate stall detector (ORIENT_STALL_ITERS,
    aborts if err_deg is not decreasing at all -- a real "stuck/oscillating/
    wrong direction" fault, not just "slow") is UNCHANGED and still protects
    against a genuinely broken run; space still stops any run manually.

14. CHANGED (2026-08-18, strategy shift): even after pt.11's real bug fix,
    the omega x lever compensation only holds the tip near its pivot to
    first order per control tick (see pt.13) -- on real hardware it cannot
    make the rotation perfectly zero-displacement, and repeated empirical
    testing confirmed 't' still drifts off the hover point by some margin.
    Rather than keep chasing an unreachable "perfect pivot" goal, 't' now:
      - calls auto_orient_about_rcm(..., compensate=False) -- pure angular
        velocity only, no linear compensation attempt at all. This doesn't
        make displacement worse (compensate=True wasn't fully preventing it
        anyway), it just stops pretending and makes the resulting motion an
        honest, simple, characterizable spin.
      - once the angle converges, proceeds STRAIGHT into TUNNEL (or plain
        INSERT outside BIPLANAR_MODE) automatically, using the tip's REAL
        post-rotation position -- no separate 'i' press needed. This is
        safe specifically because compute_goals()/do_tunnel() were already
        written (pt.10) to recompute their aim FRESH from the current tip
        position at the moment they run, never from a remembered/assumed
        hover waypoint, so whatever position 't' actually leaves the tip at
        is exactly what TUNNEL aims from -- there is no stale-target risk.
      - 'i' remains available standalone (now just calling the same
        do_tunnel() helper) to retry TUNNEL alone if the auto-chained
        attempt was stopped with space or failed, without redoing ORIENT.
    Net effect: "reach hover, rotate (whatever small drift results is
    accepted, not fought), then insert immediately" instead of insisting on
    zero-displacement rotation before allowing insertion at all. 'p'
    (PLUNGE) is UNCHANGED -- it still calls auto_orient_about_rcm with the
    default compensate=True, since that pivot (about the internal RCM,
    already inside the tissue) is a different, lower-stakes case that
    hasn't been reported as problematic.

15. FIXED (2026-08-21, real bug in the ACCEPTANCE gate, not the measurement):
    across every real vis3d attempt logged so far (2026-08-20 precision_log,
    3 valid trials / 6 ORIENT-vischeck attempts), measure_shaft_direction_3d()
    reported RANSAC inlier_ratio in the 11-38% range and was therefore
    rejected every single time by the old gate (`inlier_ratio <
    VIS3D_MIN_INLIER_RATIO=0.5`) -- the independent vision cross-check has
    NEVER once produced a usable reading. Root cause: inlier_ratio measures
    what fraction of candidate 3D points survive RANSAC, but RANSAC's whole
    purpose is to correctly recover the true line even when good points are
    a MINORITY (heavily contaminated by specular/multi-path depth dropout on
    the shiny metal shaft, see verify_depth_on_needle.py) -- a low ratio by
    itself says the SAMPLE was dirty, not that the recovered line is wrong.
    Gating on ratio alone conflates those two things, and a Monte Carlo
    replay of the real contamination regime (see
    /tmp/test_vis3d_fix.py -- gross random-outlier contamination matching
    the observed 11-38% ratios) confirms the old gate accepts essentially
    0% of trials in that regime while a fit that WOULD have been perfectly
    accurate (<2deg error) is silently discarded every time.
      NEW gate replaces the single ratio threshold with what actually
    determines trustworthiness of a RANSAC line fit under known-heavy
    contamination: (a) VIS3D_MIN_INLIERS_ABS -- enough inliers in ABSOLUTE
    count (a ratio can be low just because n_points is large; what matters
    for conditioning a 3D line is how many good points you actually have);
    (b) VIS3D_MAX_FIT_RMS_MM -- the inliers must sit TIGHTLY on the fitted
    line (well inside the 1mm RANSAC association radius, not just barely
    inside it); (c) VIS3D_MIN_EXTENT_MM (unchanged); (d) NEW --
    half_split_stability_deg(): randomly split the inlier set in half
    several times, refit each half independently by SVD, and require the
    two half-fits to agree in direction within VIS3D_MAX_STABILITY_DEG. This
    is a cheap empirical bootstrap: it directly tests REPRODUCIBILITY of the
    fitted direction (the thing we actually care about), rather than an
    indirect proxy (contamination fraction) that happens to almost never
    pass in this project's real operating conditions. inlier_ratio is still
    computed and logged (vis3d_inlier_ratio column, unchanged) as a
    diagnostic -- it is just no longer, by itself, a pass/fail gate.
    angular_uncertainty_deg is now max(old analytic atan(rms/(extent/2))
    estimate, the empirical half-split disagreement) -- so a fit that LOOKS
    tight by RMS but is not actually reproducible under resampling still
    gets an honest (larger) reported uncertainty instead of a falsely
    confident small one. The Monte Carlo test also includes a deliberately
    adversarial "ghost line" contamination model (a coherent second/
    reflection line, not just gross random noise) to confirm this change
    does not newly accept cases the old gate would not already have been
    equally fooled by -- both gates behave identically (and both remain
    genuinely fooled) in that harder synthetic case, which is an honestly
    reported limitation of RANSAC-based line fitting in general, not
    something this fix claims to solve.
      ACTION STILL REQUIRED ON THE REAL RIG (this fix only changes software
    acceptance logic -- it cannot fabricate 3D information that was never
    captured): re-run a handful of ORIENT/REORIENT cycles with
    VISION_ANGLE_CHECK=1 and confirm vis3d_ok actually starts firing True
    with a plausible n_inliers/fit_rms/stability_deg combination before
    citing any vis3d_err_deg number as a validated result.

16. CHANGED (2026-08-22, default behavior + framing): desired_insertion_dir()'s
    fixed-azimuth machinery (INSERT_AZIMUTH_DEG, added back in pt.2) already
    computed a mathematically complete TWO-AXIS frame at the insertion point
    -- (phantom normal n_hat, a fixed in-plane reference e0_hat, and their
    cross product e1_hat) -- but it was never actually exercised in the real
    2026-08-20 demos: INSERT_AZIMUTH_DEG defaulted to unset, which silently
    fell back to the LEGACY single-axis behavior (azimuth = "whatever
    direction the tool happened to be approaching from", re-derived from
    the current tip->RCM vector, not from any second fixed reference). Every
    vis3d/precision_log.csv row from that day accordingly logged
    `azimuth_deg=auto`.

    This came up directly because Prof. Iulian's 2026-08-22 group-meeting
    feedback pointed out (in general/theoretical terms, independent of this
    codebase) that a single surface-normal vector only constrains 2 of a
    rigid tool's 3 rotational DOF -- pitch/yaw (which way the shaft tilts)
    but not roll/azimuth (WHICH IN-PLANE DIRECTION it tilts toward). This is
    exactly the classical "Darboux frame" construction in differential
    geometry (surface normal + one in-plane tangent vector -> a complete,
    canonically-defined orthonormal frame, the 3rd axis being their cross
    product) -- note this uses the Darboux-frame CONSTRUCTION (normal +
    one in-plane reference -> orthonormal triad) as a static frame-at-a-
    point, not the full curve-evolving Darboux-frame theory (normal/
    geodesic curvature, relative torsion), which this project does not
    need since there is no trajectory here, just one fixed target
    orientation. For the closest real precedent of this SAME idea (full
    3D needle orientation defined via a Frenet-Serret/Darboux-style local
    frame, one axis of which must come from a source other than the
    surface/path alone), see Khadem, Rossa, Sloboda, Usmani & Tavakoli,
    "Feedback-linearization-based 3D needle steering in a Frenet-Serret
    frame using a reduced order bicycle model," ACC 2017 (U. Alberta),
    building on the foundational nonholonomic needle-steering model of
    Webster, Kim, Cowan, Chirikjian & Okamura, IJRR 2006. (An earlier
    draft of this note cited "Borgstadt/Ferreira" for this claim --
    2026-08-23 fact-check via Google Scholar/web search could not verify
    those names in this literature at all; corrected to the above, which
    were directly confirmed.) It is ALSO the exact
    quantity real vitreoretinal surgical literature cites as clinically
    significant for sutureless-sclerotomy self-sealing: e.g. the "biplanar
    5deg/30deg" and "oblique-parallel" trocar techniques (Retina Today,
    "Small-Gauge Incision Techniques") explicitly insert "parallel to the
    limbus... following the arrangement of scleral fibers" -- i.e. the
    clinically-preferred azimuth is a SPECIFIC, anatomically-referenced
    in-plane direction, not an arbitrary one.

    HONESTY NOTE (read before citing this as "clinically optimal azimuth"):
    the anatomically-correct 2nd axis would be "tangent to the LIMBUS at the
    insertion point" -- but the current phantom has no limbus feature to
    detect (see auto_target_vision_v2_limbus.py, currently shelved for that
    reason), so that specific reference is not available on this hardware.
    What IS implemented is the ENGINEERING version of the same two-axis
    idea: e0_hat is base-frame +Z (or +X as a fallback if the normal is
    near-parallel to +Z) projected into the tangent plane -- an arbitrary
    but FIXED, deterministic, reproducible reference, NOT a claim of
    anatomical/clinical optimality. This is still exactly what is needed to
    answer "how is orientation fully defined" with two named axes instead
    of one, and to make INSERT_AZIMUTH_DEG mean a specific, repeatable
    number run-to-run/position-to-position -- it just doesn't (yet) claim
    the specific choice of e0_hat is the clinically optimal one. If a real
    limbus becomes detectable on future hardware, swap e0_hat's source to
    "tangent to the limbus circle at the insertion point" (radial vector
    insertion_point-minus-limbus_center, projected into the tangent plane,
    then rotated 90deg about n_hat) -- the rest of this function's math is
    unchanged; only where e0_hat comes from would need to change.

    CHANGE: default of INSERT_AZIMUTH_DEG flipped from unset/legacy to
    "0.0" -- i.e. the deterministic two-axis frame is now the DEFAULT
    behavior, not an opt-in. Set INSERT_AZIMUTH_DEG=legacy (or =auto) to
    explicitly restore the old single-axis/approach-direction behavior
    (kept only for A/B comparison, not recommended for any run whose
    numbers you intend to report). Startup print and precision_log.csv's
    azimuth_deg column both already distinguished "auto" from a fixed
    number (no logging changes needed) -- what changes is which one you
    get with no env vars set at all.

17. FIXED (2026-08-23, two real bugs reported after the 08-20 demos):
    a) auto_move() -- used by APPROACH/TUNNEL/PLUNGE/RETRACT -- had NEITHER
       of the two anti-stall protections auto_orient_about_rcm() already had
       (pt.1's ORIENT_MIN_OMEGA floor and ORIENT_STALL_ITERS detector).
       Reported symptom: RETRACT (the one auto_move() call that travels
       farthest) sometimes hung forever, never reaching HOME, with no error
       message -- its while-loop's ONLY exit conditions were "goal reached"
       or "space/q pressed", and MAX_TRAVEL_MM's old runaway-abort safety net
       is disabled by default since pt.7. Added the same pattern: a
       MOVE_STALL_ITERS stall detector (aborts with an honest "STALLED"
       message instead of hanging if dnorm stops decreasing) and a
       MOVE_MIN_SPEED floor on the linear-taper speed (so it cannot decay
       below what real friction/backlash can overcome right at the tail of a
       long move -- the same "P-only, no floor, near a dead zone" pattern
       pt.1 already fixed for rotation, just never ported to translation).
    b) 't' (ORIENT) was reported to still visibly displace the tip off the
       hover point while rotating, despite pt.10's "self-pivot" design intent
       ("rotate about wherever the tip already is, so it shouldn't move").
       Root cause: pt.14 (2026-08-18) had set 't' to call
       auto_orient_about_rcm(..., compensate=False) -- publishing PURE
       angular velocity with v_lin forced to exactly zero. Since the tip is
       rigidly offset from the EE origin by TIP_D (~35.8mm on this rig's
       current pivot calibration), holding the EE origin exactly still while
       the tool rotates makes the (offset) tip sweep a chord of
       2*|TIP_D|*sin(angle/2) -- ~18.5mm for a 30deg ORIENT. That is a large,
       fully-predictable-in-advance systematic displacement, not the small
       stochastic residual pt.14's docstring described ("repeated empirical
       testing confirmed 't' still drifts... by some margin"); pt.14
       appears to have conflated the two. Fixed by (i) reverting 't' to
       compensate=True (the default) so the omega x lever feedforward from
       pt.11's fix -- the mathematically exact continuous-time solution for
       zero-displacement pivoting, not merely a first-order approximation --
       actually runs again, and (ii) adding a NEW closed-loop tip-position-
       HOLD trim on top of that feedforward (ORIENT_HOLD_KP, capped by
       ORIENT_HOLD_MAX_MM_S): every tick, nudge v_lin by a small term
       proportional to (hold_point - current_tip), where hold_point is the
       tip's own actual position captured once at the start of the call.
       This directly closes the loop on whatever smaller residual drift the
       feedforward's real-world imperfections (actuator lag, encoder noise,
       calibration bias -- pt.13's list) still leave behind, using the
       OBSERVED error rather than trying to model each source. 'p' (PLUNGE
       reorient) already called with compensate=True and gets the same new
       trim automatically (it was never affected by pt.14's compensate=False,
       which only touched the 't' call site).

18. NEW (2026-08-26): NEEDLE_AXIS_LOCAL override from independent vision.
    All prior NEEDLE_AXIS_LOCAL derivations normalize a POSITION vector
    (tip_inee_at_calib), which is only a valid proxy for the tool's true
    physical shaft axis if the tool mounts perfectly on-axis -- never
    independently checked before tonight. needle_axis_diagnose.py (new
    offline analysis tool) combines the vis3d_* 3D shaft-direction
    measurements pt.4 already logs with the FrameEE orientation quaternion
    (also newly logged, ee_qx/qy/qz/qw) across multiple ORIENT/REORIENT
    events at different orientations to solve for the tool's TRUE EE-frame
    axis directly, independent of tip_inee_at_calib entirely. First run
    (n=5 samples, 1 outlier excluded -- see needle_axis_diagnose.py's own
    output for the exclusion criterion) found the true axis is ~20.6deg
    away from the position-derived one, consistent across 2 azimuths and
    both ORIENT/REORIENT. See the NEEDLE_AXIS_MEASURED_20260826 override
    block (search for that name) for the full evidence writeup and caveats.
    Set NEEDLE_AXIS_OVERRIDE=0 to disable and fall back to the old
    position-derived axis (e.g. for A/B comparison of achieved
    vis3d_err_deg with vs without this correction).

Everything else (tip offset / pivot calibration loading, most of the
keyboard layout) is UNCHANGED from home_to_rcm_insert.py.

Run exactly like the original (same env vars all still work):
  source /opt/ros/melodic/setup.bash
  source /home/eyebrp/SHER_I2RIS_teleoperate/devel/setup.bash
  python3 home_to_rcm_insert_v2_precision.py

New env vars (all optional, all default to preserving old behavior/values
except the two precision fixes called out above):
  ORIENT_EASE_DEG      (default 5.0)   P-band; full ASPEED reached at/above this error
  ORIENT_MIN_OMEGA     (default 0.008) rad/s floor while err>tol (anti-stall)
  ORIENT_KI            (default 0.0006) integral gain
  ORIENT_KI_MAX_FRAC   (default 0.35)  anti-windup clamp, fraction of ASPEED
  INSERT_AZIMUTH_DEG   (default "0.0" as of pt.16, 2026-08-22 -- deterministic
                        two-axis frame; set =legacy or =auto to opt back into
                        the pre-pt.16 single-axis/approach-direction behavior)
  VISION_ANGLE_CHECK   (default 1)     independent depth-camera 3D shaft-
                                        direction cross-check (see pt.4 above)
  NEEDLE_AXIS_OVERRIDE (default 1)     use the vision-measured NEEDLE_AXIS_LOCAL
                                        correction (see pt.18); =0 to disable
  PRECISION_LOG        (default third_part/precision_log.csv)
  A_STAR_THRESH, MIN_AREA_PX, MAX_AREA_PX, MAX_RING_RESIDUAL_RATIO,
  MIN_RING_RADIUS_PX, ROI_X0/X1/Y0/Y1, RING_DIAM_MM, SMOOTH_N,
  MAX_JITTER_MM, AGREE_THRESHOLD_MM, STICKY_MAX_JUMP_PX
                        (all optional, same names/defaults as
                        auto_target_vision.py -- live 'z' ring detection)
  NORMAL_A_STAR_THRESH, NORMAL_PATCH_RADIUS_PX, NORMAL_MIN_POINTS,
  NORMAL_MAX_CENTROID_DISAGREE_MM, NORMAL_MAX_PLANE_RMS_MM
                        (all optional, same names/defaults as
                        phantom_normal_estimate.py -- live 'n' normal fit)
  VIS3D_MIN_INLIERS_ABS, VIS3D_MAX_FIT_RMS_MM, VIS3D_MAX_STABILITY_DEG,
  VIS3D_STABILITY_SPLITS
                        (all optional -- see pt.15 above, the vis3d
                        acceptance-gate fix; VIS3D_MIN_INLIER_RATIO is still
                        read but is now diagnostic-only, no longer a gate)
  MOVE_MIN_SPEED        (default 0.3)   mm/s floor for auto_move() (RETRACT/
                                          APPROACH/TUNNEL/PLUNGE), anti-stall
  MOVE_STALL_ITERS      (default 250)   ~2.5s @ 100Hz stall detector for
                                          auto_move() (see pt.17)
  ORIENT_HOLD_KP        (default 0.6)   1/s gain, closed-loop tip-hold trim
                                          on top of the omega x lever
                                          feedforward (see pt.17)
  ORIENT_HOLD_MAX_MM_S  (default 2.0)   cap on the trim term above
"""
from __future__ import print_function

# Runnable v1: relocatable source and isolated session artifacts.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from runtime_session import ROOT, data_path
import os
import re
import sys
import shutil
import time
import csv
import json
import datetime
import cv2
# 2026-10-06: found via /proc/<pid>/task/*/stat thread-level CPU breakdown
# while chasing a "tip dot can't track in real time" lag that survived
# EVERY other fix this session (origin-correction bugs, mic_live_stills
# dir-bloat throttle, mic_live_capture_ros.py memory leak, even a full
# system reboot): OpenCV's default parallel_for_ backend spins up ONE
# WORKER THREAD PER CORE (8 on this machine) and re-synchronizes them on
# every single per-frame op (cvtColor/resize/remap/aruco detect/imshow's
# own internal ops) -- confirmed live via /proc/<pid>/task listing showing
# 7 extra "python3" threads (not the main loop thread, not QXcbEventQueue)
# each independently burning CPU and growing every tick, together with
# vmstat's context-switch counter sitting at 120000-146000/sec even right
# after a clean reboot with load average otherwise under control. That
# much thread wake/sleep/synchronization churn competes directly with
# gscam/eye_robot_example/mic_live_capture_ros.py for the same 8 cores and
# effectively starves THIS process's own single-threaded render loop of
# wall-clock time between frames, independent of any one process's %CPU
# number looking "fine" in isolation. Capping OpenCV to 1 thread removes
# that self-inflicted contention entirely -- none of this script's per-
# frame image ops are large enough (640x480) to benefit from 8-way
# parallelism anyway, so there is no real throughput given up.
cv2.setNumThreads(1)
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intr_util import load_best_intrinsics  # noqa: E402
import rospy
from geometry_msgs.msg import Vector3, Transform
from std_msgs.msg import Float64MultiArray
# 2026-10-05: for the direct-ROS-subscription microscope path (MIC_ROS_LIVE),
# see _MicRosLive below -- Image is the message type gscam_decklink.launch
# already publishes on /decklink/camera/image_raw; threading.Lock guards the
# (frame, timestamp) cache a background rospy subscriber thread writes and
# the main grab_show()/draw() thread reads, same minimal-locking pattern any
# ROS-OpenCV bridge node needs (rospy callbacks run on their own thread).
import threading
import concurrent.futures
from sensor_msgs.msg import Image

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.environ.get("HANDEYE_NPZ", data_path("handeye_T.npz"))
TIP_OFFSET_NPZ = os.environ.get("TIP_OFFSET_NPZ", data_path("tip_offset.npz"))
TIP_OFFSET_PIVOT_NPZ = os.environ.get(
    "TIP_OFFSET_PIVOT_NPZ", data_path("tip_offset_pivot.npz")
)
# 2026-08-25: orientation-DEPENDENT tip model (pivot_residual_fit.py), opt-in
# via TIP_MODEL=augmented. Do NOT use "augmented" -- it scored the best
# scatter/LOO (0.11mm) of any model while placing the tip 4.54mm away from
# every other independent measurement; the pin location is a free
# parameter that a 12-param model can trade off against a constant bias,
# and LOO does not catch that. See pivot_residual_fit.py's docstring.
#
# 2026-08-25 (later same day): default changed from "rigid" to "rcmoffset".
# "rigid" reads tip_offset_pivot.npz, whose baseline is the camera-free
# CM_OFFSET_FINDER pivot fit alone -- found the same day to be biased
# ~9.4mm from the true optical tip (elastic needle flex under the pivot
# jig's force-controlled hold; see tip_inee_guard.py's CANONICAL_TIP_INEE
# docstring for the full derivation). "rcmoffset" reads
# tip_offset_rcmoffset.npz, whose tip_inee_at_calib has that correction
# applied (found via handeye_vs_tip_discriminate.py, n=28 clicks, 94.7deg
# orientation spread, 5-fold CV held-out RMS 3.86px, cross-checked via two
# independent code paths to 0.0000mm agreement, and agreeing with the
# operator's own independent manual GUI tuning to 1.43mm) while keeping
# the same orientation-dependent shape (d_const/K) rigid used. Defaulting
# to "rigid" would have silently run every future session against the
# now-known-biased baseline unless TIP_MODEL was remembered on every
# invocation -- exactly the failure mode that caused today's "green dot
# way off" symptom in the first place. Prints which model is active
# either way, so this is not a silent behavior change.
TIP_MODEL = os.environ.get("TIP_MODEL", "rcmoffset").strip().lower()
TIP_OFFSET_PIVOT_AUGMENTED_NPZ = os.environ.get(
    "TIP_OFFSET_PIVOT_AUGMENTED_NPZ", data_path("tip_offset_pivot_augmented.npz")
)
# 2026-08-25 (later same day): TIP_MODEL=rcmoffset -- see rcmoffset_fit.py's
# docstring. NOT a generic empirical fit like "augmented" above: this is the
# specific, physically-derived correction for robotModel.cpp's
# RCM2_1.RCMoffset being hardcoded to 0.0 in firmware (grep shows it is never
# set to anything else). Uses only 4 parameters (vs augmented's 12) and got
# LOO RMS 0.140mm on the same n=1203 combined set augmented got 0.110mm on --
# nearly as good, far less overfitting risk, and -- because it's the actual
# correct kinematic functional form rather than a linear-in-rotvec local
# approximation -- should not degrade when extrapolating to poses outside
# the fit range the way "augmented" can. Prefer this over "augmented" unless
# a head-to-head test on new data says otherwise.
TIP_OFFSET_RCMOFFSET_NPZ = os.environ.get(
    "TIP_OFFSET_RCMOFFSET_NPZ", data_path("tip_offset_rcmoffset.npz")
)
AUTO_TARGET_NPZ = os.environ.get("AUTO_TARGET_NPZ", data_path("auto_target.npz"))
AUTO_TARGET_MAX_AGE_S = float(os.environ.get("AUTO_TARGET_MAX_AGE_S", "5.0"))
NORMAL_NPZ = os.environ.get("NORMAL_NPZ", data_path("phantom_normal.npz"))
# 2026-10-05: append-only history of EVERY 'n' press (NORMAL_NPZ above only
# ever keeps the latest one) -- see phantom_normal_repeatability.py. Press
# 'n' several times in a row WITHOUT moving anything (phantom, scope, robot)
# to build a same-pose repeatability dataset before trusting MIC_RCM_MODE=
# circle for a real campaign.
NORMAL_SAMPLES_LOG = os.environ.get("NORMAL_SAMPLES_LOG", data_path("phantom_normal_samples.jsonl"))
INSERT_ANGLE_DEG = float(os.environ.get("INSERT_ANGLE_DEG", "30.0"))  # off the surface normal

# 2026-09-15: fixes the "TIP dot doesn't line up with the real needle tip"
# symptom. Root cause is NOT the overlay math -- it's that /FrameEE and
# /RCMPoint (robotTask's own forward kinematics) can carry a constant,
# session-wide offset from the true base frame whenever the controller's
# encoder->world reference drifts (e.g. controller left running across a
# power/CAN bus hiccup, no re-home performed -- robotTask::Home() is
# literally an unimplemented stub in EyeRobot2/code/robotTask.cpp, it logs
# "NOT IMPLEMENTED FOR THIS ROBOT YET!" and returns immediately, so neither
# the per-axis Home buttons nor the "Home Robot" button in the FLTK GUI
# actually re-establish the kinematic origin). Relative motion through
# FrameEE is still correct (confirmed: +10.5mm on retraction tracked
# correctly), only the absolute origin is shifted -- so a single constant
# 3D correction, measured directly against the camera+depth ground truth
# (same "don't trust the encoder, verify visually" principle as the
# needle-axis independent-verification work), fully cancels it. See 'c' key
# below for how this is measured; ORIGIN_CORRECTION is added to every
# /FrameEE and /RCMPoint reading at the point they're received.
ORIGIN_CORRECTION_NPZ = os.environ.get(
    "ORIGIN_CORRECTION_NPZ", data_path("origin_correction.npz")
)
# 2026-10-11: a 'c' click that lands on BACKGROUND depth (~0.6-0.8m on this
# rig, vs the needle's real ~0.1-0.2m) produces a 300-500mm ORIGIN_CORRECTION
# that makes the green TIP coincide in 2D at the click pose but then crawl
# at ~Z_true/Z_wrong of the real tip's pixel speed -- "tip travels a long
# way, green dot barely moves, and it does not catch up at rest". Cap the
# saved correction and the accepted click depth so that failure mode cannot
# silently re-enter the live control/display path.
ORIGIN_CORRECTION_MAX_MM = float(os.environ.get("ORIGIN_CORRECTION_MAX_MM", "40.0"))
ORIGIN_CLICK_Z_MIN_M = float(os.environ.get("ORIGIN_CLICK_Z_MIN_M", "0.05"))
ORIGIN_CLICK_Z_MAX_M = float(os.environ.get("ORIGIN_CLICK_Z_MAX_M", "0.35"))
ORIGIN_CLICK_DZ_MAX_M = float(os.environ.get("ORIGIN_CLICK_DZ_MAX_M", "0.012"))

# 2026-09-17: microscope->robot-base extrinsic calibration touch log (keys
# '1'-'8' as of 2026-09-24 -- widened from 4 so a held-out/LOO accuracy
# check is possible, see mic_extrinsic_calib.py --loo). See
# /home/eyebrp/Matthew/microscope_autonomy/README.md.
MIC_CALIB_LOG = os.environ.get(
    "MIC_CALIB_LOG", data_path("mic_calib_touches.jsonl")
)
# 2026-10-01 (pt.2): per-touch microscope snapshot, saved automatically by
# log_mic_calib_touch() at the moment '1'-'8' is pressed. WHY: the original
# protocol asked the operator to touch 8 points, then separately take ONE
# clean still and remember/recognize where each of those 8 points was to
# click them -- in practice (confirmed in person, 2026-10-01) the touched
# points are not always memorable/identifiable landmarks, so that later
# click step becomes an unreliable guess, which silently corrupts the
# calibration (wrong pixel paired with a real 3D point looks exactly like a
# correct point, just wrong). Fix: at the exact moment of each touch, copy
# whatever is CURRENTLY the freshest frame in MIC_LIVE_STILLS_DIR into this
# directory under the touch's own label -- the tool tip is still resting on
# the just-touched point in that frame, so it visibly marks its own
# location; nothing needs to be remembered or re-identified later. See
# click_calib_points_per_touch.py, which clicks each of these per-point
# stills instead of one shared still.
MIC_CALIB_TOUCH_STILLS_DIR = os.environ.get(
    "MIC_CALIB_TOUCH_STILLS_DIR", data_path("mic_calib_touch_stills")
)
# Written OFFLINE by mic_extrinsic_calib.py --still --mark {0,1} -- an entry
# point (one of the two confirmed candidate sclerotomy sites) already
# converted from a microscope still into robot base-frame XYZ. 'e' loads it.
MIC_ENTRY_POINT_JSON = os.environ.get(
    "MIC_ENTRY_POINT_JSON", data_path("mic_entry_point.json")
)

# 2026-09-21: one-key "live" version of the same correction, for use RIGHT
# BEFORE the final insertion move within a single trial, not just once at
# session start. See try_lock_microscope_rcm_live()'s docstring below for
# the full rationale. MIC_AUTONOMY_DIR must contain mic_perception.py and a
# fitted mic_extrinsic_calib.npz (produced once per lab session via the
# touch/click/fit protocol in /home/eyebrp/Matthew/microscope_autonomy/README.md).
# MIC_LIVE_STILLS_DIR is wherever a freshly-captured Leica still lands on
# this machine (same manual transfer path already used for every prior
# capture in this project, e.g. USB/network copy into "Lecia Microscope/").
MIC_AUTONOMY_DIR = os.environ.get(
    "MIC_AUTONOMY_DIR", str(ROOT / "microscope_autonomy")
)
MIC_EXTRINSIC_CALIB_NPZ = os.environ.get(
    "MIC_EXTRINSIC_CALIB_NPZ", data_path('mic_extrinsic_calib.npz')
)
MIC_LIVE_STILLS_DIR = os.environ.get(
    "MIC_LIVE_STILLS_DIR", os.path.join(MIC_AUTONOMY_DIR, "mic_live_stills")
)
# Post-PLUNGE microscope ground-truth stills (needle still embedded), one set per trial.
# Trial number = number of trials already in this dir + 1, so it survives restarts.
TRIAL_STILLS_DIR = os.environ.get("TRIAL_STILLS_DIR", data_path("trial_plunge_stills"))
TRIAL_STILLS_N = int(os.environ.get("TRIAL_STILLS_N", "3"))
TRIAL_STILLS_TIMEOUT_S = float(os.environ.get("TRIAL_STILLS_TIMEOUT_S", "5.0"))
# 2026-10-05: the user's correction -- the microscope feed is a genuine
# real-time video stream (DeckLink SDI capture card, 1920x1080@59.94,
# published to ROS by gscam_decklink.launch), NOT a manually-captured single
# still. Until now this script only ever consumed it through
# mic_live_capture_ros.py's disk relay (throttled to interval=0.5s i.e. 2Hz,
# plus a JPEG encode/write/read/decode round trip, plus get_mic_preview()'s
# os.listdir() directory poll every frame) -- a real, avoidable bottleneck
# between a ~60Hz source and whatever rate this script could actually track
# marks/the circle target at. MIC_ROS_LIVE=1 subscribes DIRECTLY to
# MIC_ROS_TOPIC in-process (see _MicRosLive below), the exact same pattern
# already used for D405 frames via pipe.wait_for_frames() -- no disk I/O,
# no polling, no 0.5s throttle, just whatever rate frames actually arrive
# at (capped by MIC_ROS_PROCESS_INTERVAL_S purely to bound mic_perception.py's
# own CPU cost per frame, not by any artificial relay). Defaults to OFF
# (falls back to the disk-polling path unchanged) so a lab session where
# mic_live_capture_ros.py is already running, or the ROS topic genuinely
# isn't up, loses nothing -- this is purely additive, preferred when live.
MIC_ROS_LIVE = os.environ.get("MIC_ROS_LIVE", "0") == "1"
MIC_ROS_TOPIC = os.environ.get("MIC_ROS_TOPIC", "/decklink/camera/image_raw")
MIC_ROS_PROCESS_INTERVAL_S = float(os.environ.get("MIC_ROS_PROCESS_INTERVAL_S", "0.05"))  # 20Hz cap
MIC_ROS_STALE_S = float(os.environ.get("MIC_ROS_STALE_S", "0.5"))  # no new frame in 0.5s -> fall back to disk

# 2026-10-07: DEFAULT FLIPPED from "1" to "0" at the user's explicit
# decision (most lab sessions are free-space/calibration work -- hand-eye
# recal, microscope touch calibration, etc. -- with no real trocar/RCM to
# protect yet, so "0", no pivot-compensation drift during manual rotation
# jogs, is now the common case). This default got silently re-enabled
# (back to the old "1") on each of the last two days' script restarts
# simply because nobody remembered to pass the env var every time --
# exactly the kind of silent, hard-to-notice regression the user asked to
# stop happening. If/when a session actually has a real trocar/sclerotomy
# inserted and needs the old rotate-about-live-RCM protection back, set
# MANUAL_JOG_RCM_PIVOT=1 explicitly for that session.
MANUAL_JOG_RCM_PIVOT = os.environ.get("MANUAL_JOG_RCM_PIVOT", "0") == "1"


class _MicRosLive(object):
    """2026-10-05: in-process subscriber to the microscope's live ROS Image
    topic -- see MIC_ROS_LIVE's comment above for why this replaces the
    disk-relay path when available. Decoding reuses
    mic_live_capture_ros.decode_image_msg() (imported, NOT copy-pasted --
    see this project's own history of >=2 separate default-path copy-paste
    bugs for why that caution is a standing rule here, not paranoia) so a
    future encoding-format fix to that function covers this path too,
    automatically, with zero risk of the two drifting apart.

    Thread safety: rospy.Subscriber callbacks run on a background thread;
    .get_latest() is read from the main grab_show()/draw() thread. The lock
    only ever protects a plain dict assignment/read (microseconds), so it
    cannot meaningfully stall either thread."""

    def __init__(self, topic):
        self.topic = topic
        self._lock = threading.Lock()
        self._frame = None
        self._t = 0.0
        self._n_recv = 0
        self._decode_err_streak = 0
        self._sub = None

    def start(self):
        sys.path.insert(0, MIC_AUTONOMY_DIR)
        from mic_live_capture_ros import decode_image_msg
        self._decode = decode_image_msg
        self._sub = rospy.Subscriber(self.topic, Image, self._cb, queue_size=1, buff_size=2**24)
        print("  MIC_ROS_LIVE: subscribed to %s (in-process, no disk relay)" % self.topic)

    def _cb(self, msg):
        frame = self._decode(msg)
        if frame is None:
            self._decode_err_streak += 1
            if self._decode_err_streak in (1, 10, 100) or self._decode_err_streak % 1000 == 0:
                print("  !! MIC_ROS_LIVE: decode_image_msg failed (streak=%d, enc=%s) -- "
                      "falling back to disk relay until this recovers" %
                      (self._decode_err_streak, msg.encoding))
            return
        self._decode_err_streak = 0
        with self._lock:
            self._frame = frame
            self._t = time.time()
            self._n_recv += 1

    def get_latest(self):
        """Returns (frame, t, n_recv) -- frame is None if nothing received
        yet, or if the last message failed to decode and no good frame has
        ever arrived."""
        with self._lock:
            return self._frame, self._t, self._n_recv


# Instantiated (if MIC_ROS_LIVE=1) once rospy.init_node() has run inside
# main() -- rospy.Subscriber requires a node to already be initialized, so
# this cannot be created at import time like the dict above; see main()'s
# startup block.
_mic_ros_live = None
# 2026-09-24: DELIBERATELY a separate file from MIC_CALIB_LOG above, written
# by a separate key ('9', not '1'-'8'). This is a structural (not just
# disciplinary) guarantee that independent-verification campaign points can
# never accidentally end up inside the file mic_extrinsic_calib.py --fit
# reads -- i.e. a held-out point can never silently leak into the training
# set, same principle as split_pairs_holdout.py's train/val separation. See
# mic_independent_verify.py + MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md.
MIC_CAMPAIGN_LOG = os.environ.get(
    "MIC_CAMPAIGN_LOG", os.path.join(MIC_AUTONOMY_DIR, "mic_campaign_touches.jsonl")
)
MIC_TARGET_MARK = int(os.environ.get("MIC_TARGET_MARK", "0"))
# 2026-10-05: the user's explicit architectural correction -- autonomous
# targeting should NOT be "pick which of the 2 literal ink dots", it should
# be "compute a point on the circle concentric with the limbus" (mark0/mark1
# are just 2 arbitrary calibration landmarks on that same circle, not the
# clinically meaningful target itself). This mirrors real vitreoretinal
# surgery, where the sclerotomy site is planned as a clock-position +
# distance-from-limbus, not "whichever of 2 ink dots happens to be there".
# mark0/mark1 detection in mic_perception.py is UNCHANGED and still runs
# every frame (still used for calibration: it is what empirically establishes
# the limbus-to-entry RADIUS as a position-invariant anatomical constant --
# see entry_radius_from_marks(); verified 2026-10-05 on 6 clean touch stills:
# mark0 radius-from-limbus std=3.1px/mean=517.7px i.e. <1% relative spread,
# INCLUDING across a mid-session phantom reposition -- strong evidence this
# radius is a real geometric invariant, not a position-dependent accident).
# MIC_RCM_MODE="mark" (default) is the exact pre-2026-10-05 behavior: 'y'
# targets whichever literal ink mark MIC_TARGET_MARK points at, zero change.
# MIC_RCM_MODE="circle" makes 'y' instead target a SYSTEM-COMPUTED point at
# MIC_RCM_ANGLE_DEG around the limbus (radius self-calibrated per-frame from
# whichever marks ARE visible that frame via entry_radius_from_marks(), or
# fixed via MIC_RCM_RADIUS_PX if set) -- i.e. "any point on the circle", not
# restricted to exactly 2 options, with the angle itself being a clinical/
# protocol decision made ONCE per experiment design, not a per-frame choice
# between 2 ink blobs.
MIC_RCM_MODE = os.environ.get("MIC_RCM_MODE", "mark")
assert MIC_RCM_MODE in ("mark", "circle"), "MIC_RCM_MODE must be 'mark' or 'circle', got %r" % MIC_RCM_MODE
MIC_RCM_ANGLE_DEG = os.environ.get("MIC_RCM_ANGLE_DEG")
MIC_RCM_ANGLE_DEG = float(MIC_RCM_ANGLE_DEG) if MIC_RCM_ANGLE_DEG is not None else None
if MIC_RCM_MODE == "circle" and MIC_RCM_ANGLE_DEG is None:
    raise SystemExit("MIC_RCM_MODE=circle requires MIC_RCM_ANGLE_DEG to be set "
                      "(the system-chosen clock-position angle, degrees, atan2(dy,dx) "
                      "convention around the limbus center) -- no silent default, this "
                      "is a protocol/clinical decision, not something to guess.")
MIC_RCM_RADIUS_PX = os.environ.get("MIC_RCM_RADIUS_PX")
MIC_RCM_RADIUS_PX = float(MIC_RCM_RADIUS_PX) if MIC_RCM_RADIUS_PX is not None else None
# Per-trial target angle so successive trials puncture fresh, separately measurable sites:
# trial N uses entry (N-1) % len, applied when its HOME is latched ('h'). Empty = fixed angle.
MIC_RCM_ANGLE_SCHEDULE = [float(s) for s in os.environ.get("MIC_RCM_ANGLE_SCHEDULE", "").split(",") if s.strip()]
INSERT_ANGLE_SCHEDULE = [float(s) for s in os.environ.get("INSERT_ANGLE_SCHEDULE", "").split(",") if s.strip()]
MIC_FRAME_W, MIC_FRAME_H = 1920, 1080
MIC_TARGET_EDGE_MARGIN_PX = float(os.environ.get("MIC_TARGET_EDGE_MARGIN_PX", "60"))
# 2026-10-11: circle-mode D405 estimate = validated ring->mark offset + the constant
# mark->target step on the mark's own limbus circle (pixel step mapped through the
# extrinsic's linear part). Valid while the phantom keeps its training yaw
# (PHANTOM_MAX_YAW_DEG check). "tangent" = legacy phantom-normal local-frame model.
MIC_CIRCLE_D405_MODEL = os.environ.get("MIC_CIRCLE_D405_MODEL", "fixed")
assert MIC_CIRCLE_D405_MODEL in ("fixed", "tangent")
# mark0 polar position around the plug centre, 3 post-retract stills of trial 2 (2026-10-11)
MIC_NOMINAL_MARK_THETA_DEG = float(os.environ.get("MIC_NOMINAL_MARK_THETA_DEG", "nan"))
MIC_NOMINAL_MARK_R_PX = float(os.environ.get("MIC_NOMINAL_MARK_R_PX", "nan"))
if os.path.isfile(data_path("ring_to_entry_offset.npz")):
    with np.load(data_path("ring_to_entry_offset.npz"), allow_pickle=False) as _nom:
        if "mark0_theta_deg" in _nom and "mark0_radius_px" in _nom:
            MIC_NOMINAL_MARK_THETA_DEG = float(_nom["mark0_theta_deg"])
            MIC_NOMINAL_MARK_R_PX = float(_nom["mark0_radius_px"])

# Must match mic_perception's MIC_TARGET_ANGLE_REL_MARK: MIC_RCM_ANGLE_DEG is then an offset
# from the measured mark angle (rotation-invariant target).
MIC_ANGLE_REL_MARK = os.environ.get("MIC_TARGET_ANGLE_REL_MARK", "") != ""
# RCM z for '`': "d405plane" = phantom face height at the target from the D405 ring plane
# (follows tilt/height changes); "mic" = the extrinsic's fixed z_plane_mm.
MIC_Z_SOURCE = os.environ.get("MIC_Z_SOURCE", "d405plane")
assert MIC_Z_SOURCE in ("d405plane", "mic")
MIC_Z_NORMAL_MAX_AGE_MM = float(os.environ.get("MIC_Z_NORMAL_MAX_AGE_MM", "3.0"))
# 2026-10-05: MIC_RCM_RADIUS_PX above only overrides the PIXEL-space radius
# used by the authoritative 'y'/'e' microscope path (mic_perception.
# point_on_plug_circle). The D405-ring-based APPROACH/HOVER-ONLY paths
# ('x' and '`') use a SEPARATE 3D metric radius (radius_local_mm_mark{0,1},
# tangent-plane-local-frame, fit per-mark by mic_ring_offset_verify.py
# --fit-circle from live ring+microscope samples -- see try_lock_vision_rcm_
# with_offset()/try_correspondence_check() below). On 2026-10-05, mark0 and
# mark1's independently-fit radius_local disagreed by ~5mm (14.0 vs 19.2mm,
# >30% relative) -- i.e. today's empirical fit does NOT support "any point
# on the circle shares one radius" as a clean geometric invariant, so
# picking mark0's (lower-LOO-error) value by default is an arbitrary
# tie-break, not a validated choice. If you have a real clinically-specified
# fixed radius (limbus-to-entry distance per protocol, NOT fit from noisy
# session data), set MIC_RCM_RADIUS_LOCAL_MM and it will be used verbatim
# instead of EITHER mark's npz-derived value, in both 'x' and '`' -- same
# "no silent default, this is a protocol/clinical decision" spirit as
# MIC_RCM_ANGLE_DEG above. Leave unset to keep the existing (mark0-by-
# default) empirical-fit behavior unchanged.
MIC_RCM_RADIUS_LOCAL_MM = os.environ.get("MIC_RCM_RADIUS_LOCAL_MM")
MIC_RCM_RADIUS_LOCAL_MM = (float(MIC_RCM_RADIUS_LOCAL_MM)
                            if MIC_RCM_RADIUS_LOCAL_MM is not None else None)
# 2026-09-28: per-frame mark0/mark1 pixel centroids from mic_perception.py
# have NO temporal smoothing at all -- unlike _ring_live's D405 path (which
# requires RING_SMOOTH_N consistent samples + jitter<RING_MAX_JITTER_MM
# before calling itself "LOCKED"), every new microscope still just overwrote
# the on-screen mark position outright. Measured live (2026-09-28): mark0
# centroid has ~3.7px RMS / ~9px max radial jitter frame-to-frame even under
# good, unoccluded conditions -- visible on screen as the target mark
# "flying around". _mic_mark_live below adds the SAME kind of rolling
# average + jitter + lock-gate the ring already has, so 'y' can require an
# actually-stable mark instead of trusting whatever the newest single frame
# happened to say.
MIC_MARK_SMOOTH_N = int(os.environ.get("MIC_MARK_SMOOTH_N", "6"))
MIC_MARK_MAX_JITTER_PX = float(os.environ.get("MIC_MARK_MAX_JITTER_PX", "4.0"))
# 2026-09-21: staleness gate was disabled by default (1e12s -- effectively
# "never stale") for demo/reporting use with the old manual-still-transfer
# workflow, where an old still was fine to preview/lock deliberately.
#
# 2026-09-24: RE-ENABLED with a real value now that DeckLink+gscam+
# mic_live_capture_ros.py feed this folder continuously (interval=0.5s) --
# leaving it disabled is exactly what let a genuinely dead pipeline hide in
# plain sight: mic_live_capture_ros.py silently failed every single write
# for ~43 hours (2026-09-22 18:12 -> 2026-09-24 13:13, see that file's
# cv2.imencode bugfix) while this gate's `fresh = age_s <= 1e12` stayed
# True the entire time, so both the 'y' key and the PiP panel's color
# coding kept reporting "fresh" against a 2-day-old frozen frame. 5
# seconds gives a few missed 0.5s-interval frames of slack (a genuinely
# live feed should never fall behind by more than 1-2s) without letting a
# truly dead writer hide for hours. Still overridable via env var if a
# slower capture cadence is ever used.
MIC_LIVE_MAX_AGE_S = float(os.environ.get("MIC_LIVE_MAX_AGE_S", "5"))

# 2026-09-21: fixed base-frame vector from "wherever 'z' (D405 red-ring
# detector) locks on THIS phantom" (the limbus -- see chat write-up
# 2026-09-21) to "the true ink-marked entry point" (from the microscope),
# learned ONCE via mic_extrinsic_calib.py --ring-offset. 'x' below adds it
# to the live 'z' lock, so APPROACH/HOVER can use D405 alone and land much
# closer to the real target than raw ring-center -- still not a substitute
# for 'y'/'e' right before the actual insertion move.
RING_TO_ENTRY_OFFSET_NPZ = os.environ.get(
    "RING_TO_ENTRY_OFFSET_NPZ", data_path('ring_to_entry_offset.npz')
)
# 2026-10-05: output of mic_ring_offset_verify.py --fit-circle -- the
# phantom-pose-invariant, tangent-plane-local-frame generalization of the
# fixed vector above (radius_local_mm/theta_local_deg instead of a raw
# base-frame XYZ offset -- see tangent_frame_from_normal() in
# mic_extrinsic_calib.py). Only read when MIC_RCM_MODE=="circle".
RING_TO_ENTRY_OFFSET_CIRCLE_NPZ = os.environ.get(
    "RING_TO_ENTRY_OFFSET_CIRCLE_NPZ", data_path('ring_to_entry_offset_circle.npz')
)
# 2026-09-28: append-only log of PAIRED (D405 ring-lock, microscope
# entry-mark) samples, written by the '0' key (log_ring_offset_sample()).
# mic_ring_offset_verify.py --fit consumes this to learn+validate
# RING_TO_ENTRY_OFFSET_NPZ with leave-one-out cross-validation instead of
# mic_extrinsic_calib.py --ring-offset's old single-snapshot fit (which had
# zero held-out validation -- see mic_ring_offset_verify.py's docstring).
RING_OFFSET_SAMPLES_LOG = os.environ.get(
    "RING_OFFSET_SAMPLES_LOG", data_path("ring_offset_samples.jsonl")
)

# 2026-09-21: SHIFT+H records one hand-eye (ArUco-on-shaft) calibration
# sample IN THIS SAME running instance, so jogging (wasd/rf/kmjl/uo) and
# hand-eye sample recording can interleave without the handeye_capture_one.py
# toggle dance (quit this script to free the camera, jog, quit, capture,
# relaunch...). Writes the EXACT SAME CSV schema handeye_calibrate.py's own
# interactive loop uses, so that script's "Resumed N pairs" auto-load + 's'
# to solve still works unmodified once you're done collecting samples here.
HANDEYE_DATA_DIR = os.environ.get("HANDEYE_DATA_DIR", data_path("handeye_data"))
HANDEYE_PAIRS_CSV = os.environ.get("HANDEYE_PAIRS_CSV", os.path.join(HANDEYE_DATA_DIR, "pairs.csv"))
HANDEYE_SAMPLE_N_TARGET = int(os.environ.get("HANDEYE_SAMPLE_N_TARGET", "12"))
sys.path.insert(0, HERE)
try:
    from handeye_calibrate import detect_aruco as _detect_aruco
    _ARUCO_IMPORT_ERR = None
except Exception as _e:
    _detect_aruco = None
    _ARUCO_IMPORT_ERR = str(_e)


def calib_freshness_problems(phase="use"):
    """Every artifact whose numbers were computed THROUGH R_he/t_he (or
    through the microscope's current physical pose) must be newer than the
    hand-eye it is combined with. Returns a list of human-readable problems;
    empty list == safe to log campaign data. Missing files are reported too."""
    probs = []
    he = os.path.realpath(NPZ) if 'NPZ' in globals() else None
    he_t = os.path.getmtime(he) if he and os.path.isfile(he) else None
    if he_t is None:
        return ["handeye_T.npz missing"]
    deps = [("phantom_normal", NORMAL_NPZ, "press 'n'"),
            ("mic_extrinsic_calib", MIC_EXTRINSIC_CALIB_NPZ, "15-touch refit"),
            (("ring_to_entry_offset_circle", RING_TO_ENTRY_OFFSET_CIRCLE_NPZ,
              "'`'/'0' samples + mic_ring_offset_verify.py --fit-circle")
             if (MIC_RCM_MODE == "circle" and MIC_CIRCLE_D405_MODEL == "tangent") else
             ("ring_to_entry_offset", RING_TO_ENTRY_OFFSET_NPZ,
              "'0' samples + mic_ring_offset_verify.py --fit"))]
    if phase == "collect":
        deps = [d for d in deps if not d[0].startswith("ring_to_entry_offset")]
    elif phase != "use":
        raise ValueError("unknown calibration phase")
    for name, path, how in deps:
        rp = os.path.realpath(path)
        if not os.path.isfile(rp):
            probs.append("%s MISSING (%s)" % (name, how))
        elif os.path.getmtime(rp) < he_t:
            probs.append("%s OLDER than handeye_T.npz (%s)" % (name, how))
    if float(np.linalg.norm(ORIGIN_CORRECTION)) > ORIGIN_CORRECTION_MAX_MM:
        probs.append("ORIGIN_CORRECTION |%.0f|mm too large" % np.linalg.norm(ORIGIN_CORRECTION))
    return probs


def load_origin_correction(path):
    if not os.path.isfile(path):
        print("Origin correction: %s not found -- using zero correction "
              "(press 'c' in the live window to measure one)." % path)
        return np.zeros(3)
    z = np.load(path)
    corr = np.asarray(z["correction"], np.float64)
    mag = float(np.linalg.norm(corr))
    ts = float(z["timestamp"]) if "timestamp" in z else None
    age_s = (time.time() - ts) if ts else None
    z_click = float(z["depth_m"]) if "depth_m" in z.files else None
    bad_mag = mag > ORIGIN_CORRECTION_MAX_MM
    bad_z = (z_click is not None and
             (z_click < ORIGIN_CLICK_Z_MIN_M or z_click > ORIGIN_CLICK_Z_MAX_M))
    if bad_mag or bad_z:
        print("Origin correction: REJECTING %s -- |corr|=%.1fmm%s%s. "
              "This is the background-depth / far-Z click failure mode "
              "(green TIP aligns in 2D at one pose, then barely moves). "
              "Using zero. Press 'c' only on a click whose depth is the "
              "NEEDLE (%.0f-%.0fmm), or SHIFT+C to keep zero." % (
                  path, mag,
                  (" (limit %.1fmm)" % ORIGIN_CORRECTION_MAX_MM) if bad_mag else "",
                  ("  click depth=%.0fmm" % (z_click * 1000.0)) if z_click is not None else "",
                  ORIGIN_CLICK_Z_MIN_M * 1000.0, ORIGIN_CLICK_Z_MAX_M * 1000.0))
        return np.zeros(3)
    print("Origin correction: %s OK  correction=[%.2f %.2f %.2f] mm%s" % (
        path, corr[0], corr[1], corr[2],
        ("  (measured %.0f min ago -- re-measure with 'c' if the controller "
         "has been restarted since)" % (age_s / 60.0)) if age_s is not None else ""))
    return corr

_az_env = os.environ.get("INSERT_AZIMUTH_DEG", "0.0").strip()
# pt.16 (2026-08-22): default CHANGED from "" (legacy auto-azimuth) to "0.0"
# (deterministic fixed-frame azimuth) -- see module docstring pt.16. Set
# INSERT_AZIMUTH_DEG=legacy (or =auto) to explicitly opt back into the old
# "whatever direction the tool happened to approach from" behavior, e.g. for
# an A/B comparison against this fix.
INSERT_AZIMUTH_DEG = None if _az_env.lower() in ("legacy", "auto", "none") else float(_az_env)

ORIENT_TOL_DEG = float(os.environ.get("ORIENT_TOL_DEG", "0.3"))
ORIENT_TIMEOUT_S = float(os.environ.get("ORIENT_TIMEOUT_S", "0"))  # <=0 disables (2026-08-18, pt.13)
ORIENT_STALL_ITERS = int(os.environ.get("ORIENT_STALL_ITERS", "250"))  # ~2.5s @ 100Hz
ORIENT_EASE_DEG = float(os.environ.get("ORIENT_EASE_DEG", "5.0"))
ORIENT_MIN_OMEGA = float(os.environ.get("ORIENT_MIN_OMEGA", "0.008"))
ORIENT_KI = float(os.environ.get("ORIENT_KI", "0.0006"))
ORIENT_KI_MAX_FRAC = float(os.environ.get("ORIENT_KI_MAX_FRAC", "0.35"))
# pt.18 (2026-09-28): max staleness (s) for the /JointPosition (q3) feed before
# auto_orient_about_rcm()'s achievable-plane projection falls back to the old
# unprojected behavior -- see that function's docstring pt.18 for why this
# fix exists (root cause of "wrong CW/CCW direction, gets stuck" reports).
ORIENT_Q3_MAX_AGE_S = float(os.environ.get("ORIENT_Q3_MAX_AGE_S", "0.5"))
VISION_ANGLE_CHECK = os.environ.get("VISION_ANGLE_CHECK", "1") != "0"
# --- measure_shaft_direction_3d() conditioning thresholds (see docstring) ---
VIS3D_ROI_RADIUS_PX = int(os.environ.get("VIS3D_ROI_RADIUS_PX", "90"))
VIS3D_MIN_POINTS = int(os.environ.get("VIS3D_MIN_POINTS", "8"))
VIS3D_MIN_EXTENT_MM = float(os.environ.get("VIS3D_MIN_EXTENT_MM", "3.0"))
VIS3D_DEPTH_WIN_PX = int(os.environ.get("VIS3D_DEPTH_WIN_PX", "4"))
VIS3D_SAMPLE_STEP_PX = float(os.environ.get("VIS3D_SAMPLE_STEP_PX", "3.0"))
VIS3D_ANGLE_PRIOR_TOL_DEG = float(os.environ.get("VIS3D_ANGLE_PRIOR_TOL_DEG", "22.0"))
VIS3D_ANGLE_MERGE_DEG = float(os.environ.get("VIS3D_ANGLE_MERGE_DEG", "8.0"))
# RANSAC (not plain least-squares) line fit -- see verify_depth_on_needle.py
# and the 2026-08-18 review notes: depth failure on a thin/metallic shaft is
# expected to produce OUTLIERS (multi-path / specular dropout), not just
# extra Gaussian noise, so a plain SVD fit (no outlier rejection) is the
# WRONG tool here even though it is mathematically valid on clean data.
VIS3D_RANSAC_THRESH_MM = float(os.environ.get("VIS3D_RANSAC_THRESH_MM", "1.0"))
VIS3D_RANSAC_ITERS = int(os.environ.get("VIS3D_RANSAC_ITERS", "300"))
# VIS3D_MIN_INLIER_RATIO: kept for backward-compat env-var support and still
# LOGGED (vis3d_inlier_ratio), but as of pt.15 (2026-08-21) it is no longer a
# hard accept/reject gate -- see the acceptance-gate constants below and the
# module docstring pt.15 for why a ratio-only gate was wrong.
VIS3D_MIN_INLIER_RATIO = float(os.environ.get("VIS3D_MIN_INLIER_RATIO", "0.5"))
# --- pt.15 (2026-08-21) replacement acceptance gate: absolute inlier count +
# fit tightness + resampling-based reproducibility, instead of inlier ratio.
VIS3D_MIN_INLIERS_ABS = int(os.environ.get("VIS3D_MIN_INLIERS_ABS", "15"))
VIS3D_MAX_FIT_RMS_MM = float(os.environ.get("VIS3D_MAX_FIT_RMS_MM", "0.5"))
VIS3D_MAX_STABILITY_DEG = float(os.environ.get("VIS3D_MAX_STABILITY_DEG", "5.0"))
VIS3D_STABILITY_SPLITS = int(os.environ.get("VIS3D_STABILITY_SPLITS", "5"))
_VIS3D_RNG = np.random.RandomState(12345)  # fixed seed: reproducible RANSAC across runs
_VIS3D_STABILITY_RNG = np.random.RandomState(20260821)  # separate stream for half-split resampling
PRECISION_LOG = os.environ.get("PRECISION_LOG", data_path("precision_log.csv"))
# pt.22 (2026-09-28): per-tick ORIENT diagnostic trace, see auto_orient_
# about_rcm()'s docstring/_flush_orient_trace() -- overwritten (not
# appended) on every ORIENT call, so it always holds the MOST RECENT attempt
# for inspection after the fact.
ORIENT_TRACE_CSV = os.environ.get("ORIENT_TRACE_CSV", data_path("orient_trace_last.csv"))

# --- live, in-process RCM ring detection -- same env var names/defaults as
# auto_target_vision.py, see module docstring pt.6 ---
RING_A_STAR_THRESH = int(os.environ.get("A_STAR_THRESH", "145"))
RING_MIN_AREA_PX = float(os.environ.get("MIN_AREA_PX", "1200"))
RING_MAX_AREA_PX = float(os.environ.get("MAX_AREA_PX", "20000"))
RING_MAX_RESIDUAL_RATIO = float(os.environ.get("MAX_RING_RESIDUAL_RATIO", "0.32"))
RING_MIN_RADIUS_PX = float(os.environ.get("MIN_RING_RADIUS_PX", "20"))
RING_ROI_X0 = int(os.environ.get("ROI_X0", "100"))
RING_ROI_X1 = int(os.environ.get("ROI_X1", "540"))
RING_ROI_Y0 = int(os.environ.get("ROI_Y0", "200"))
RING_ROI_Y1 = int(os.environ.get("ROI_Y1", "480"))
RING_DIAM_MM = float(os.environ.get("RING_DIAM_MM", "0"))
RING_SMOOTH_N = int(os.environ.get("SMOOTH_N", "15"))
RING_MAX_JITTER_MM = float(os.environ.get("MAX_JITTER_MM", "3.0"))
RING_AGREE_THRESHOLD_MM = float(os.environ.get("AGREE_THRESHOLD_MM", "3.0"))
RING_STICKY_MAX_JUMP_PX = float(os.environ.get("STICKY_MAX_JUMP_PX", "80"))
RING_DEPTH_WIN_PX = int(os.environ.get("RING_DEPTH_WIN_PX", "5"))
# 2026-09-28 (pt.19, real bug -- reported symptom: "ring not visible" showing
# up often once the microscope's top-down light/the tool itself is near the
# phantom): traced against real recorded frames from THIS session
# (recordings/home_rcm_v2_20260928_160527.mp4) -- confirmed the failure mode
# in the concrete cases found there is the ring's red annulus getting
# fragmented into >=2 disconnected pieces (a thin occluding line/specular
# gap crossing it) whose INDIVIDUAL areas each fall under RING_MIN_AREA_PX
# even though they are clearly the same ring together (e.g. one real frame:
# pieces of 1098px when 1200px was required). A bigger closing kernel
# bridges small gaps like this (validated: raised the closest real failure
# frame's largest-component area from 1098px to 2365px at kernel=17, still
# annulus-shaped per the residual-ratio check). NOT a fix for the tool
# physically blocking a large arc of the ring outright (e.g. directly
# hovering over it) -- that is a genuine, physically-inherent limit of a
# tool-co-mounted camera sharing its view with the very tool it's tracking
# near, which no color/morphology tuning can recover; _ring_live's existing
# sticky-lock design (see update_live_ring_detection()) already handles
# THAT correctly by holding the last good lock rather than dropping it.
RING_CLOSE_KERNEL_PX = int(os.environ.get("RING_CLOSE_KERNEL_PX", "11"))
# Fallback for when NO single connected component passes on its own: try
# merging pairs of nearby, non-trivial components (e.g. the two halves of a
# ring split by one thin occluding line) into one candidate before giving
# up. Distance/size bounds keep this from merging two UNRELATED red things
# that just happen to both be in the ROI.
RING_MERGE_MAX_DIST_PX = float(os.environ.get("RING_MERGE_MAX_DIST_PX", "150"))
RING_MERGE_MIN_PIECE_PX = float(os.environ.get("RING_MERGE_MIN_PIECE_PX", "150"))

# 2026-09-28 (pt.20, real bug -- "ring is not visible" happening often,
# reported by user as correlated with the microscope's top-down light):
# diagnosed live on 2026-09-28 16:42 via screenshot of the actual running
# session -- the ring's own on-screen status line read "RING NOT VISIBLE
# (1150 red px, largest component area=532px, need 1200)" while the ring
# was VISUALLY present but washed to a pale pink (not deep red/maroon).
# Measured LAB a* directly on that live frame: the pink ring's a* had
# collapsed to ~124-129, indistinguishable from the black background's
# a*~126 -- i.e. NOT a fragmentation/occlusion problem (pt.19's fix does
# not apply here), but a genuine loss of a*-channel contrast under strong
# top-down illumination. Controlled simulation (blend a known-good real
# ring frame f_007.png toward white, i.e. synthetic top-light washout)
# confirms the mechanism AND finds a fix: at blend=0.7 (severe washout)
# a*'s ring-vs-background gap collapses from +46 to +13.2 (why the a*
# threshold fails), while HSV Saturation's gap only drops from +84.2 to
# +32.5 (still a clean ~4x ring/background ratio) -- Saturation is far
# more lighting-invariant than a* because it normalizes by brightness.
# Fix: detect_ring() now falls back to an Otsu-adaptive threshold on the
# Saturation channel (within the same ROI) whenever the a*-based pass
# (pt.19's own close+merge logic included) finds nothing. Otsu is clamped
# to [RING_SAT_OTSU_MIN, RING_SAT_OTSU_MAX] to avoid the degenerate case
# where the ROI has no genuine bimodal red/background split at all (e.g.
# truly nothing there), which would otherwise let Otsu pick a near-zero
# threshold that swallows the whole ROI as "candidate".
RING_USE_SAT_FALLBACK = int(os.environ.get("RING_USE_SAT_FALLBACK", "1"))
RING_SAT_OTSU_MIN = float(os.environ.get("RING_SAT_OTSU_MIN", "30"))
RING_SAT_OTSU_MAX = float(os.environ.get("RING_SAT_OTSU_MAX", "200"))

# 2026-10-01 (pt.23, camera-side counterpart to pt.20's Saturation-fallback
# software workaround): the microscope's top-down light was still driving
# the D405 color sensor's AUTO exposure up near ~20000, which broadly clips
# the white phantom and washes out the ring well before pt.20's fallback
# even gets a chance to run. A controlled exposure sweep (manual AE off,
# fixed exposure/gain, see tests_camera_exposure/) found that fixing
# Exposure=12000, Gain=16 substantially reduces that broad overexposure
# while leaving ring detection, depth validity, 3D RCM lock, and the
# phantom-normal estimator all validated as unaffected (750/750 ring
# accept, 100% valid depth, lock at frame 15 w/ 0.267mm jitter; 50/50
# normal estimates, mean plane RMS ~0.64mm) -- see
# tests_camera_exposure/D405_Camera_Settings_AI_Integration_Handoff.md for
# the full validation record. This is a camera-config change only: it does
# NOT touch the detector thresholds/fallback logic above (pt.19/pt.20),
# hand-eye, pivot, origin, or any other calibration file.
D405_AUTO_EXPOSURE = int(os.environ.get("D405_AUTO_EXPOSURE", "0"))
D405_EXPOSURE = float(os.environ.get("D405_EXPOSURE", "12000"))
D405_GAIN = float(os.environ.get("D405_GAIN", "16"))

# --- live, in-process phantom-normal estimation -- same env var
# names/defaults as phantom_normal_estimate.py, see module docstring pt.6 ---
NORMAL_A_STAR_THRESH = int(os.environ.get("NORMAL_A_STAR_THRESH", "145"))
NORMAL_PATCH_RADIUS_PX = int(os.environ.get("NORMAL_PATCH_RADIUS_PX", "15"))
NORMAL_MIN_POINTS = int(os.environ.get("NORMAL_MIN_POINTS", "80"))
NORMAL_MAX_CENTROID_DISAGREE_MM = float(os.environ.get("NORMAL_MAX_CENTROID_DISAGREE_MM", "5.0"))
NORMAL_MAX_CENTROID_DEPTH_MM = float(os.environ.get("NORMAL_MAX_CENTROID_DEPTH_MM", "10.0"))
NORMAL_MAX_PLANE_RMS_MM = float(os.environ.get("NORMAL_MAX_PLANE_RMS_MM", "1.0"))

VEL_TOPIC = os.environ.get("VEL_TOPIC", "/SHER21/eyerobot2/desiredTipVelocities")
ANG_TOPIC = os.environ.get("ANG_TOPIC", "/SHER21/eyerobot2/desiredTipVelocitiesAngular")
FRAMEEE_TOPIC = os.environ.get("FRAMEEE_TOPIC", "/SHER21/eye_robot/FrameEE")
RCMPOINT_TOPIC = os.environ.get("RCMPOINT_TOPIC", "/SHER21/eye_robot/RCMPoint")
# 2026-09-28: needed by auto_orient_about_rcm()'s achievable-rotation-plane
# projection fix -- see that function's docstring pt.18 for the full story.
# mtsDoubleVec RobotState.JointPosition() (5-DoF: q0..q4, SAME array/units fed
# into robotModel::ComputeJacobianTip()), published verbatim as this topic's
# Float64MultiArray.data -- confirmed via SHER_I2RIS_teleoperate/src/eye_robot/
# Example/main.cpp ("GetJointPositionPub") and robotTask.cpp line ~1158
# (JointPositionPub.Assign(RobotState.JointPosition())). q3 = data[3], radians
# (robotModel.cpp ComputeActuatorToJointPosition: outJointVal[3] =
# inActuatorVal[4] * 0.0005 * cmnPI_180 -- degrees->radians conversion).
JOINTPOS_TOPIC = os.environ.get("JOINTPOS_TOPIC", "/SHER21/eye_robot/JointPosition")
RECORD_DIR = os.environ.get("OVERLAY_RECORD_DIR", data_path("recordings"))

SPEED = float(os.environ.get("JOG_SPEED", "3.0"))
SPEED_MAX = float(os.environ.get("JOG_MAX", "8.0"))
ASPEED = float(os.environ.get("JOG_ASPEED", "0.02"))
ASPEED_MAX = float(os.environ.get("JOG_AMAX", "0.3"))
HOVER_MM = float(os.environ.get("HOVER_MM", "8.0"))
INSERT_DEPTH_MM = float(os.environ.get("INSERT_DEPTH_MM", "4.5"))  # only used if BIPLANAR_MODE=0
# Bi-planar clinical dimensions per the CIS II final report's "Automated
# Bi-Planar Insertion Kinematics": 1.5mm oblique scleral tunnel (at
# INSERT_ANGLE_DEG off-normal) + 3.0mm vertical vitreous plunge (straight
# along the normal, re-pivoted about a NEW internal RCM at the tunnel end).
TUNNEL_DEPTH_MM = float(os.environ.get("TUNNEL_DEPTH_MM", "1.5"))
PLUNGE_DEPTH_MM = float(os.environ.get("PLUNGE_DEPTH_MM", "3.0"))
BIPLANAR_MODE = os.environ.get("BIPLANAR_MODE", "1") != "0"
RETRACT_PAUSE_S = float(os.environ.get("RETRACT_PAUSE_S", "1.0"))  # cannula-stabilization pause before 'b'
MAX_TRAVEL_MM = float(os.environ.get("MAX_TRAVEL_MM", "0"))  # <=0 == disabled (2026-08-18: was aborting normal HOME->hover/insert travel mid-move; set e.g. MAX_TRAVEL_MM=60 to re-enable a runaway-motion safety cap if you want one back)
POS_TOL_MM = float(os.environ.get("POS_TOL_MM", "0.4"))
# pt.17 (2026-08-23): auto_move() (APPROACH/TUNNEL/PLUNGE/RETRACT) had NEITHER
# of the two anti-stall protections auto_orient_about_rcm() already has (see
# pt.1/ORIENT_MIN_OMEGA and ORIENT_STALL_ITERS above) -- reported symptom was
# RETRACT hanging forever, never reaching HOME, with no error message at all.
# See pt.17 changelog entry (bottom of file) for the full reasoning.
# TUNNEL/PLUNGE run at a constant PIERCE_SPEED, slowing only in the last PIERCE_TAPER_MM:
# the generic 3mm linear taper made a 1.5mm tunnel crawl at <=1.5 -> 0.3 mm/s, which dented
# and laterally dragged the phantom surface before puncture (2026-10-11).
PIERCE_SPEED = float(os.environ.get("PIERCE_SPEED", "3.0"))
PIERCE_TAPER_MM = float(os.environ.get("PIERCE_TAPER_MM", "0.5"))
PIERCE_MOVES = ("TUNNEL", "PLUNGE")
# Slow constant-speed withdrawal: at 3 mm/s the silicone gripped the needle and the
# unclamped phantom was lifted/dragged off its seat (trial 3, 2026-10-11).
RETRACT_AXIAL_SPEED = float(os.environ.get("RETRACT_AXIAL_SPEED", "1.0"))
# After the slow axial withdraw, keep lifting along the normal (normal SPEED) to this height above
# the internal RCM before the lateral move to HOME, so the tip cannot sweep across the dome
# (trial 1 2026-10-12: globe rotated ~10 mm in its holder at the start of RETRACT).
RETRACT_CLEAR_MM = float(os.environ.get("RETRACT_CLEAR_MM", "20.0"))
MOVE_MIN_SPEED = float(os.environ.get("MOVE_MIN_SPEED", "0.3"))  # mm/s floor while err>tol (anti-stall, mirrors ORIENT_MIN_OMEGA)
MOVE_STALL_ITERS = int(os.environ.get("MOVE_STALL_ITERS", "250"))  # ~2.5s @ 100Hz, mirrors ORIENT_STALL_ITERS
# pt.17 also re-enables the omega x lever pivot compensation for 't' (reverted
# pt.14's compensate=False) and adds a small closed-loop position-hold trim on
# top of it, to actually fix (not just detect) the "ORIENT displaces the tip
# off the hover point" symptom -- see pt.17 for the quantified reasoning.
ORIENT_HOLD_KP = float(os.environ.get("ORIENT_HOLD_KP", "0.6"))  # 1/s, P-gain trimming residual hold-point drift
ORIENT_HOLD_MAX_MM_S = float(os.environ.get("ORIENT_HOLD_MAX_MM_S", "2.0"))  # cap on the trim term alone
PULSE_S = float(os.environ.get("PULSE_S", "0.20"))
RECORD = os.environ.get("RECORD", "1") != "0"

COLOR_TIP = (130, 220, 120)
COLOR_RCM = (70, 165, 250)
COLOR_HOME = (240, 190, 90)
COLOR_HOVER = (110, 225, 235)
COLOR_SHAFT = (235, 235, 235)
COLOR_TARGET = (220, 130, 235)
COLOR_VISCHK = (255, 255, 0)

_ee = {"p": None, "q": None, "t": None}
_rcm = {"p": None}
# 2026-09-28 (pt.18): q3 (radians) from the live /SHER21/eye_robot/JointPosition
# feed -- see JOINTPOS_TOPIC comment. "t" is the wall-clock time of the last
# received message, used by auto_orient_about_rcm() to detect a stale/dead
# feed (falls back to unprojected/legacy behavior with a one-time warning
# rather than silently using a stale q3, which would project onto the WRONG
# achievable plane -- worse than not projecting at all).
_robot_q3 = {"val": None, "t": 0.0}
ORIGIN_CORRECTION = load_origin_correction(ORIGIN_CORRECTION_NPZ)
_vision_rcm = {"p": None, "t": 0.0, "n": 0, "jitter": None}
_target_dir_cache = {"dir": None, "lock_t": None, "azimuth_deg": None}
_ring_live = {"last_center": None, "hist_depth": [], "hist_size": [], "det": False,
              "cx": None, "cy": None, "r_px": None, "smoothed": None, "jitter": None,
              "n": 0, "locked": False, "status_msg": "ring detection not started yet",
              "depth_src": None}
# 2026-09-28: mirrors _ring_live's rolling-average/jitter/lock pattern, but
# for the microscope's mark0/mark1 pixel centroids (see MIC_MARK_SMOOTH_N
# above). Keyed by mark index (0/1) so both marks are tracked independently
# regardless of which one MIC_TARGET_MARK currently points at.
_mic_mark_live = {"hist": {0: [], 1: []}, "smoothed": {0: None, 1: None},
                   "jitter_px": {0: None, 1: None}, "n": {0: 0, 1: 0},
                   "locked": {0: False, 1: False},
                   "status_msg": "microscope mark tracking not started yet"}
# 2026-10-05: mirrors _mic_mark_live's rolling-average/jitter/lock pattern
# EXACTLY, but for the single system-computed point on the limbus-concentric
# circle (rec["target_cx"/"target_cy"], see mic_perception.point_on_plug_circle
# + MIC_RCM_MODE/MIC_RCM_ANGLE_DEG above). Only populated/consulted when
# MIC_RCM_MODE=="circle"; left untouched (all None/empty) in the default
# "mark" mode so there is zero behavior change unless explicitly opted in.
_mic_target_live = {"hist": [], "smoothed": None, "jitter_px": None, "n": 0,
                     "locked": False,
                     "status_msg": "circle-target tracking not started yet"}
# 2026-10-05: latest result of the '`' correspondence-check key, held purely
# for the on-screen HUD line (see draw()) -- mutated IN PLACE (dict.update(),
# never rebound with '=') by try_correspondence_check(), matching every
# other shared-state dict in this module (_ring_live, _vision_rcm,
# _mic_mark_live above) so no `global` declaration is needed anywhere.
_last_correspondence = {"t": None, "disc_raw_mm": None, "disc_corr_mm": None, "offset_source": None}
# 2026-09-21: live ArUco-on-shaft status for the hand-eye recording feature
# (SHIFT+H) -- updated every grab_show() tick purely for the on-screen "is
# the marker visible right now" readout; record_handeye_sample() does its
# OWN fresh burst-average detection when actually pressed, it does not read
# this cache (staleness here would only affect the status text, never a
# saved sample).
_aruco_live = {"found": False, "center": None, "label": None,
               "status_msg": "hand-eye ArUco detection not started yet"}


def kasa_circle_fit(xs, ys):
    """Algebraic least-squares circle fit -- ported verbatim from
    auto_target_vision.py. Works on scattered/broken-arc points (no closed
    contour needed), which matters because specular highlights knock chunks
    out of the phantom's red rim every frame."""
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    Amat = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(Amat, b, rcond=None)
    cx, cy, c = sol
    r2 = c + cx ** 2 + cy ** 2
    if r2 <= 0:
        return None
    return cx, cy, float(np.sqrt(r2))


def ring_residual_ratio(xs, ys, cx, cy, r):
    """RMS distance from the fitted circle / r -- ported verbatim from
    auto_target_vision.py. Separates a true thin annulus (low ratio) from a
    solid blob / lighting false-positive (high ratio), which raw area alone
    cannot do."""
    d = np.hypot(xs - cx, ys - cy)
    resid = np.sqrt(np.mean((d - r) ** 2))
    return resid / max(r, 1e-6)


def _ring_try_candidates(cand_list, last_center):
    """Shared inner loop: given a list of (area, xs, ys) candidates already
    sorted best-first, try each against the sticky-jump/radius/shape checks
    and return (cx, cy, r, n_px, None) on the first pass, else
    (None, None, None, None, worst_reason_seen)."""
    worst_reason = None
    for area, xs, ys in cand_list:
        fit = kasa_circle_fit(xs, ys)
        if fit is None:
            worst_reason = "circle fit degenerate"
            continue
        cx, cy, r = fit
        if last_center is not None:
            d = np.hypot(cx - last_center[0], cy - last_center[1])
            if d > RING_STICKY_MAX_JUMP_PX:
                worst_reason = "center jumped %.0fpx > STICKY_MAX_JUMP_PX=%.0f" % (d, RING_STICKY_MAX_JUMP_PX)
                continue
        if r < RING_MIN_RADIUS_PX:
            worst_reason = "fitted r=%.1fpx < MIN_RING_RADIUS_PX=%.1f" % (r, RING_MIN_RADIUS_PX)
            continue
        ratio = ring_residual_ratio(xs, ys, cx, cy, r)
        if ratio > RING_MAX_RESIDUAL_RATIO:
            worst_reason = ("shape residual ratio=%.3f > MAX_RING_RESIDUAL_RATIO=%.2f "
                             "(not annulus-shaped enough -- try raising this env var)" %
                             (ratio, RING_MAX_RESIDUAL_RATIO))
            continue
        return cx, cy, r, len(xs), None
    return None, None, None, None, worst_reason


def _segment_ring_from_mask(mask, last_center, chan_label):
    """Shared pt.19/pt.20 segmentation body: given an already-thresholded
    (0/255) full-frame binary mask (ROI already applied, rest zero), run
    close->open->connected-components->candidate-select->merge-fallback and
    return (cx, cy, r, n_px, diag). Extracted out of detect_ring() (pt.20)
    so the same pipeline can be tried on more than one channel's mask
    (a* first, then Saturation as a lighting-robust fallback) without
    duplicating the close/merge logic. chan_label is just for diagnostics."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((RING_CLOSE_KERNEL_PX, RING_CLOSE_KERNEL_PX), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_red_px = int(np.count_nonzero(mask))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None, None, None, None, ("no %s-passing pixels in ROI at all" % chan_label)

    candidates = []
    best_area = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        best_area = max(best_area, area)
        if area < RING_MIN_AREA_PX or area > RING_MAX_AREA_PX:
            continue
        candidates.append((area, i))
    candidates.sort(reverse=True)

    cand_pixels = [(area, *np.nonzero(labels == i)[::-1]) for area, i in candidates]  # (area, xs, ys)
    cx, cy, r, n_px, reason = _ring_try_candidates(cand_pixels, last_center)
    if cx is not None:
        return cx, cy, r, n_px, None

    # pt.19 fallback: no single component (in [MIN_AREA_PX, MAX_AREA_PX] on
    # its own) passed -- try pairing up NEARBY smaller pieces (below
    # RING_MIN_AREA_PX individually, e.g. the two halves of a ring split by
    # one thin occluding line/glare gap) into single merged candidates.
    small_pieces = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= RING_MERGE_MIN_PIECE_PX]
    merged_cands = []
    for a_idx in range(len(small_pieces)):
        for b_idx in range(a_idx + 1, len(small_pieces)):
            i, j = small_pieces[a_idx], small_pieces[b_idx]
            d = np.hypot(centroids[i][0] - centroids[j][0], centroids[i][1] - centroids[j][1])
            if d > RING_MERGE_MAX_DIST_PX:
                continue
            merged_area = stats[i, cv2.CC_STAT_AREA] + stats[j, cv2.CC_STAT_AREA]
            if merged_area < RING_MIN_AREA_PX or merged_area > RING_MAX_AREA_PX:
                continue
            ys_i, xs_i = np.nonzero(labels == i)
            ys_j, xs_j = np.nonzero(labels == j)
            xs = np.concatenate([xs_i, xs_j]); ys = np.concatenate([ys_i, ys_j])
            merged_cands.append((merged_area, xs, ys))
    merged_cands.sort(key=lambda c: -c[0])
    cx, cy, r, n_px, reason2 = _ring_try_candidates(merged_cands, last_center)
    if cx is not None:
        return cx, cy, r, n_px, None

    if not candidates and not merged_cands:
        return None, None, None, None, ("%s: %d px passed, largest component area=%d px "
                                         "(need %d-%d, even after merging nearby pieces)" %
                                         (chan_label, n_red_px, best_area,
                                          RING_MIN_AREA_PX, RING_MAX_AREA_PX))
    return None, None, None, None, ("%s: %s" % (chan_label, reason2 or reason or
                                     "no candidate passed shape check (tried merging nearby fragments too)"))


def configure_d405_manual_exposure(profile, auto_exposure=D405_AUTO_EXPOSURE,
                                    exposure=D405_EXPOSURE, gain=D405_GAIN):
    """2026-10-01 (pt.23): apply the validated fixed D405 exposure/gain so the
    microscope's top-down light stops driving auto-exposure into broad
    overexposure on the white phantom -- see tests_camera_exposure/ for the
    sweep that picked these defaults. MUST be called right after
    pipe.start(cfg) and BEFORE the warm-up frames (frames grabbed before the
    manual settings land would still reflect the old auto-exposure state).
    Finds the sensor supporting rs.option.exposure (prefers the one whose
    name contains "Stereo", matching the D405's exposure-capable module),
    disables auto exposure, applies the requested exposure/gain, reads the
    values back from the device (never assumes set_option() was honored
    exactly), and prints them so a mismatch is visible immediately instead
    of silently proceeding under the wrong imaging state. Does not touch
    ring/normal detector thresholds or any calibration .npz file.
    """
    device = profile.get_device()
    exposure_sensor = None
    for sensor in device.query_sensors():
        try:
            name = sensor.get_info(rs.camera_info.name)
        except Exception:
            name = ""
        if sensor.supports(rs.option.exposure):
            if exposure_sensor is None:
                exposure_sensor = sensor
            if "Stereo" in name:
                exposure_sensor = sensor
                break
    if exposure_sensor is None:
        print("WARNING: no D405 sensor supporting manual exposure was found; "
              "running under whatever auto-exposure state the camera started in.")
        return

    if exposure_sensor.supports(rs.option.enable_auto_exposure):
        exposure_sensor.set_option(rs.option.enable_auto_exposure, float(auto_exposure))
    if exposure_sensor.supports(rs.option.exposure):
        exposure_sensor.set_option(rs.option.exposure, float(exposure))
    if exposure_sensor.supports(rs.option.gain):
        exposure_sensor.set_option(rs.option.gain, float(gain))

    actual_ae = (exposure_sensor.get_option(rs.option.enable_auto_exposure)
                 if exposure_sensor.supports(rs.option.enable_auto_exposure) else float("nan"))
    actual_exposure = (exposure_sensor.get_option(rs.option.exposure)
                       if exposure_sensor.supports(rs.option.exposure) else float("nan"))
    actual_gain = (exposure_sensor.get_option(rs.option.gain)
                   if exposure_sensor.supports(rs.option.gain) else float("nan"))
    print("D405 imaging settings: auto_exposure=%.0f exposure=%.0f gain=%.0f "
          "(requested auto_exposure=%d exposure=%.0f gain=%.0f)" %
          (actual_ae, actual_exposure, actual_gain, auto_exposure, exposure, gain))
    if (abs(actual_exposure - exposure) > 1.0) or (abs(actual_gain - gain) > 1.0):
        print("WARNING: D405 actual exposure/gain did not match the requested values -- "
              "inspect the RealSense configuration before trusting ring/normal detection.")


def detect_ring(color_bgr, last_center=None):
    """2D ring (RCM marker) detection -- ported from auto_target_vision.py's
    detect_ring(), using this file's RING_* module constants instead of
    that script's own module-level ones (same names/defaults). Returns
    (cx, cy, r_px, n_pixels, diag) or (None, None, None, None, diag); diag
    explains WHY nothing passed.

    FIXED (2026-09-28, pt.19): see RING_CLOSE_KERNEL_PX/RING_MERGE_*
    comments above for the real-frame evidence motivating (a) a bigger
    closing kernel (bridges small gaps -- partial occlusion/specular
    dropout splitting the ring into 2 pieces each individually under
    RING_MIN_AREA_PX) and (b) a merge-nearby-components fallback tried only
    when no single component passes on its own. Deliberately does NOT try
    to fix the tool-directly-occluding-a-large-arc case -- see that
    constant's comment for why no perception tuning can, and why the
    existing sticky lock in update_live_ring_detection() already covers it.

    FIXED (2026-09-28, pt.20): see RING_USE_SAT_FALLBACK comments above for
    the real-frame + controlled-simulation evidence that a* alone loses
    ring/background contrast under strong top-down light (a genuinely
    different failure mode than pt.19's occlusion/fragmentation). Now
    falls back to an Otsu-adaptive threshold on HSV Saturation (within the
    same ROI) when the a*-based pass finds nothing at all.
    """
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    a_full = lab[:, :, 1]
    h, w = a_full.shape
    x0, x1 = max(0, RING_ROI_X0), min(w, RING_ROI_X1)
    y0, y1 = max(0, RING_ROI_Y0), min(h, RING_ROI_Y1)
    a_mask = np.zeros((h, w), dtype=np.uint8)
    a_sub = (a_full[y0:y1, x0:x1] > RING_A_STAR_THRESH).astype(np.uint8) * 255
    a_mask[y0:y1, x0:x1] = a_sub

    cx, cy, r, n_px, reason_a = _segment_ring_from_mask(a_mask, last_center, "a*>%d" % RING_A_STAR_THRESH)
    if cx is not None:
        return cx, cy, r, n_px, None

    if not RING_USE_SAT_FALLBACK:
        return None, None, None, None, reason_a

    # SAFETY GATE (pt.20, found during my own validation): Saturation is
    # hue-agnostic -- unlike a*, which is red/magenta-biased, ANY strongly
    # colored object in the ROI (not just red/pink ones) can pass a
    # Saturation threshold. Tested in isolation (no last_center) on real
    # frames, this occasionally locked onto an unrelated saturated object
    # far from the true ring. The existing STICKY_MAX_JUMP_PX check (below,
    # inside _segment_ring_from_mask -> _ring_try_candidates) reliably
    # rejects those once a real lock is established (verified: re-running
    # those exact false-positive frames WITH a realistic last_center, all
    # were correctly rejected as ">80px jump"). But with last_center=None
    # (the very first-ever detection this session, before any lock
    # exists) that guard cannot run -- so the Saturation fallback is only
    # allowed to REACQUIRE/MAINTAIN an already-established lock through a
    # lighting-washout event, never to originate a brand new one. If the
    # very first 'z' press happens to land on washed-out lighting, this
    # will correctly report "not visible" rather than risk a bad first
    # lock; get the a* pathway a first hit (e.g. wait for/adjust lighting)
    # before it can take over.
    if last_center is None:
        return None, None, None, None, "%s (Sat fallback skipped: no established lock yet to sticky-check against)" % reason_a

    # pt.20: a* found nothing -- try an Otsu-adaptive threshold on HSV
    # Saturation within the same ROI. Clamp Otsu's pick so a genuinely
    # empty/no-red ROI can't degrade into "half the ROI is a candidate".
    s_full = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    s_roi = s_full[y0:y1, x0:x1]
    otsu_th, _ = cv2.threshold(s_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_th = float(np.clip(otsu_th, RING_SAT_OTSU_MIN, RING_SAT_OTSU_MAX))
    s_mask = np.zeros((h, w), dtype=np.uint8)
    s_mask[y0:y1, x0:x1] = (s_roi > otsu_th).astype(np.uint8) * 255

    cx, cy, r, n_px, reason_s = _segment_ring_from_mask(s_mask, last_center, "Sat>%.0f(Otsu)" % otsu_th)
    if cx is not None:
        return cx, cy, r, n_px, None

    return None, None, None, None, "%s | %s" % (reason_a, reason_s)


def save_target_atomic(path, base_xyz, base_xyz_pnp, diff_mm, agree, n, jitter_mm):
    """Write auto_target.npz via temp-file + os.replace -- ported verbatim
    from auto_target_vision.py's save_target_atomic() (a non-atomic np.savez
    got caught mid-write by an external pkill once and corrupted this file
    for a REAL run, so this is not defensive-programming theater)."""
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "wb") as f:
        np.savez(f, base_xyz=base_xyz, base_xyz_pnp=base_xyz_pnp, base_xyz_depth=base_xyz,
                  diff_mm=diff_mm, agree=agree, n=n, jitter_mm=jitter_mm, t=time.time())
    os.replace(tmp, path)


def load_auto_target(path):
    if not os.path.isfile(path):
        return None
    try:
        z = np.load(path)
        d = {"p": np.asarray(z["base_xyz"], float), "t": float(z["t"]),
             "n": int(z["n"]), "jitter": float(z["jitter_mm"])}
        if "diff_mm" in z.files:
            d["diff_mm"] = float(z["diff_mm"])
            d["agree"] = bool(z["agree"])
        return d
    except Exception:
        return None


def _cb_ee(m):
    _ee["p"] = np.array([m.translation.x, m.translation.y, m.translation.z], np.float64) \
        + ORIGIN_CORRECTION
    _ee["q"] = np.array([m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w], np.float64)
    # 2026-10-06 (pt.3): stamp the wall-clock time this callback actually
    # ran, NOT the message's own header (Transform has no header/stamp) --
    # lets grab_show() show how stale cur_tip()'s input really is, to tell
    # apart "main loop is slow" from "/FrameEE callback itself is starved"
    # as the cause of any observed TIP-dot lag, instead of guessing.
    _ee["t"] = time.time()


def _cb_rcm(m):
    _rcm["p"] = np.array([m.x, m.y, m.z], np.float64) + ORIGIN_CORRECTION


def _cb_jointpos(m):
    """pt.18 (2026-09-28): m.data is RobotState.JointPosition() verbatim (5
    elements, q0..q4) -- see JOINTPOS_TOPIC comment for the C++-side proof.
    Only q3 (index 3, radians) is needed downstream, but guard against a
    malformed/short message rather than raising out of a ROS callback."""
    if len(m.data) > 3:
        _robot_q3["val"] = float(m.data[3])
        _robot_q3["t"] = time.time()


def load_handeye(path):
    if not os.path.isfile(path):
        return None, None
    z = np.load(path)
    return z["R"], z["t"]


def load_tip_offset(path):
    if not os.path.isfile(path):
        return np.zeros(3), False
    z = np.load(path)
    w = np.asarray(z["w"], float) if "w" in z.files else np.zeros(3)
    return w, True


def load_tip_offset_pivot(path):
    """Returns (d, ok, quality_dict). quality_dict surfaces the diagnostics
    pivot_calibration.py already computed and saved (fit_rms/loo_rms/
    orientation_spread_deg/age) but which used to be silently discarded here
    -- this offset feeds BOTH cur_tip() (position) AND NEEDLE_AXIS_LOCAL
    (the orientation axis auto_orient_about_rcm() converges to), so a bad
    pivot calibration would bias the achieved 3D angle AND the RCM-insertion
    point TOGETHER, in a way the software's own error readout cannot see
    (see README's "补充5" root-cause discussion, 2026-08-18)."""
    if not os.path.isfile(path):
        return None, False, None
    z = np.load(path)
    if "d" not in z.files:
        return None, False, None
    q = {}
    if "fit_rms" in z.files:
        q["fit_rms_mm"] = float(z["fit_rms"])
    if "loo_rms" in z.files:
        loo = float(z["loo_rms"])
        if np.isfinite(loo):
            q["loo_rms_mm"] = loo
    if "orientation_spread_deg" in z.files:
        q["spread_deg"] = float(z["orientation_spread_deg"])
    if "n" in z.files:
        q["n"] = int(z["n"])
    if "tip_inee_at_calib" in z.files:
        q["tip_inee_at_calib"] = np.asarray(z["tip_inee_at_calib"], float)
    try:
        q["age_hr"] = (time.time() - os.path.getmtime(path)) / 3600.0
    except OSError:
        pass
    return np.asarray(z["d"], float), True, q


def load_tip_offset_pivot_augmented(path):
    """Returns (d0, C, R_mean, ok, quality_dict) for the orientation-dependent
    model from pivot_residual_fit.py: tip_offset(R) = d0 + C @ rotvec(R_mean.T@R).
    See that script's docstring for why this model exists (rigid `d` alone
    was independently shown POOR by tip_pin_check_v2.py on 2026-08-25)."""
    if not os.path.isfile(path):
        return None, None, None, False, None
    z = np.load(path)
    if "d0" not in z.files or "C" not in z.files or "R_mean" not in z.files:
        return None, None, None, False, None
    q = {}
    for key in ("loo_rms_augmented", "loo_rms_baseline", "lam", "fit_rms_combined_mm"):
        if key in z.files:
            v = z[key]
            v = float(v) if v.shape == () else v
            if not (isinstance(v, float) and not np.isfinite(v)):
                q[key] = v
    if "n" in z.files:
        q["n"] = int(z["n"])
    if "tip_inee_at_calib" in z.files:
        tip_inee = np.asarray(z["tip_inee_at_calib"], float)
        if np.all(np.isfinite(tip_inee)):
            q["tip_inee_at_calib"] = tip_inee
    return (np.asarray(z["d0"], float), np.asarray(z["C"], float),
            np.asarray(z["R_mean"], float), True, q)


def load_tip_offset_rcmoffset(path):
    """Returns (d_const, K, ok, quality_dict) for the physically-derived
    model from rcmoffset_fit.py: tip_offset(R) = d_const + K*[0,sin(q4),cos(q4)]
    where q4 = atan2(-R[1,2], R[1,1]) is the roll angle recovered from R
    under the firmware's own R=Ry(q3)@Rx(q4) 2-DOF-wrist kinematics. See
    rcmoffset_fit.py's docstring for the full derivation and validation."""
    if not os.path.isfile(path):
        return None, None, False, None
    z = np.load(path)
    if "d_const" not in z.files or "K" not in z.files:
        return None, None, False, None
    q = {}
    for key in ("loo_rms_mm", "loo_rms_rigid_mm"):
        if key in z.files:
            v = float(z[key])
            if np.isfinite(v):
                q[key] = v
    if "n" in z.files:
        q["n"] = int(z["n"])
    if "tip_inee_at_calib" in z.files:
        tip_inee = np.asarray(z["tip_inee_at_calib"], float)
        if np.all(np.isfinite(tip_inee)):
            q["tip_inee_at_calib"] = tip_inee
    return np.asarray(z["d_const"], float), float(z["K"]), True, q


def load_phantom_normal(path):
    if not os.path.isfile(path):
        return None
    z = np.load(path)
    return {"point": np.asarray(z["point_base"], float),
            "normal": np.asarray(z["normal_base"], float),
            "n_points": int(z["n_points"]), "rms": float(z["plane_rms_mm"])}


def rotation_between(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    c = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    axis = np.cross(v_from, v_to)
    s = np.linalg.norm(axis)
    angle = np.arctan2(s, c)
    if s < 1e-9:
        if c > 0:
            return np.zeros(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(v_from[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, perp)
        axis /= np.linalg.norm(axis)
        return axis * np.pi
    axis = axis / s
    return axis * angle


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def base_to_pixel(p_base, R, t, intr):
    """BASE-frame point -> pixel, via the camera-frame point and the RealSense
    SDK's own forward projection (applies intr's distortion model), NOT a
    naive zero-distortion pinhole formula.

    2026-08-23 finding: hand-eye (handeye_reproj_refine.py) is fit by
    minimizing cv2.projectPoints(..., K, dist) reprojection error -- i.e. R,t
    were solved WITH distortion in the loop. The old pinhole-only version of
    this function ignored distortion entirely, so overlay error was small
    near the image center (where distortion ~0) but grew toward the edges --
    exactly the "tip lines up at some home positions, not others" symptom,
    since different home positions put the tip at different pixel locations.
    rs2_project_point_to_pixel is the exact forward counterpart of
    deproject_mm()'s rs2_deproject_pixel_to_point() below, so projection and
    deprojection are now symmetric for whichever intrinsics (self-cal or
    factory) load_best_intrinsics() picked.
    """
    c = R.T.dot(p_base - t)
    if c[2] <= 1e-3:
        return None
    u, v = rs.rs2_project_point_to_pixel(intr, [float(c[0]), float(c[1]), float(c[2])])
    return (int(round(u)), int(round(v)))


def deproject_mm(intr, px, py, z_m):
    """Pixel + depth(meters) -> 3D point (mm) in the CAMERA frame."""
    pt = rs.rs2_deproject_pixel_to_point(intr, [float(px), float(py)], float(z_m))
    return np.array(pt, dtype=np.float64) * 1000.0


def needle_depth_m_at_click(depth_f, u, v, z_min=None, z_max=None, max_win=15):
    """Nearest plausible NEEDLE depth (meters) near a clicked pixel.

    2026-10-11: a 5x5 min() of *any* valid stereo hit still returns the
    table/wall when the thin needle itself has no depth. Accepting that
    far Z is what produced the 300-500mm ORIGIN_CORRECTION and the
    "tip moves a long way, green dot barely moves" overlay.
    """
    if z_min is None:
        z_min = ORIGIN_CLICK_Z_MIN_M
    if z_max is None:
        z_max = ORIGIN_CLICK_Z_MAX_M
    any_z = []
    for win in (2, 5, 10, max_win):
        zs = []
        for dv in range(-win, win + 1):
            for du in range(-win, win + 1):
                try:
                    z = float(depth_f.get_distance(int(u + du), int(v + dv)))
                except Exception:
                    continue
                if z > 0:
                    any_z.append(z)
                if z_min < z < z_max:
                    zs.append(z)
        if zs:
            return float(min(zs)), win, len(zs), None
    far = float(min(any_z)) if any_z else None
    return None, max_win, 0, far


def depth_min_at(depth_mm, px, py, win):
    """MIN depth in mm (not mean/median) in a small window -- correct choice
    for a THIN foreground object (the needle shaft) against a background: a
    window average is biased toward whichever is more numerous (usually
    background), min picks out the nearest surface, i.e. the shaft itself.
    Same reasoning already used by auto_target_vision.py's depth_at()."""
    h, w = depth_mm.shape
    x0, x1 = max(0, px - win), min(w, px + win + 1)
    y0, y1 = max(0, py - win), min(h, py + win + 1)
    v = depth_mm[y0:y1, x0:x1]
    v = v[v > 0]
    if v.size == 0:
        return float("nan")
    return float(np.min(v))


# The current phantom's red ring surrounds a through-hole, so the depth AT the
# ring centre is noisy stereo inside the hole; depth_min_at() there picks a
# near-side outlier (2026-10-11: D405 ring read ~16mm nearer than the microscope
# plug centre, offset within 18deg of the camera axis). "ring" mode instead takes
# the median depth of the red ring's own pixels, which for a planar ring equals
# the plane depth at its centre. RING_DEPTH_MODE=center restores the old behaviour.
RING_DEPTH_MODE = os.environ.get("RING_DEPTH_MODE", "ring")
RING_DEPTH_MIN_PX = int(os.environ.get("RING_DEPTH_MIN_PX", "30"))


RING_DEPTH_ANNULUS = (float(os.environ.get("RING_DEPTH_R_IN", "0.65")),
                      float(os.environ.get("RING_DEPTH_R_OUT", "1.10")))
RING_DEPTH_RANSAC_MM = float(os.environ.get("RING_DEPTH_RANSAC_MM", "1.5"))
RING_DEPTH_MIN_INLIER_FRAC = float(os.environ.get("RING_DEPTH_MIN_INLIER_FRAC", "0.35"))
# 'n' normal source. "ring" = plane of the red ring (RANSAC, averaged over NORMAL_RING_FRAMES
# frames). "bowl" = legacy 15px patch at the ring centre, which lands inside the through-hole
# and gave 18-45deg spurious tilts on 2026-10-11 (ring plane: 0-6deg on the same frames).
NORMAL_MODE = os.environ.get("NORMAL_MODE", "ring")
assert NORMAL_MODE in ("ring", "bowl"), "NORMAL_MODE must be 'ring' or 'bowl'"
NORMAL_RING_FRAMES = int(os.environ.get("NORMAL_RING_FRAMES", "10"))
NORMAL_RING_MAX_SPREAD_DEG = float(os.environ.get("NORMAL_RING_MAX_SPREAD_DEG", "8.0"))
PHANTOM_MAX_TILT_DEG = float(os.environ.get("PHANTOM_MAX_TILT_DEG", "10.0"))
# mark0->mark1 direction (base XY) of the 2026-10-11 n=8 ring-offset training set: 34.0-36.9deg.
PHANTOM_TRAIN_YAW_DEG = float(os.environ.get("PHANTOM_TRAIN_YAW_DEG", "nan"))
PHANTOM_MAX_YAW_DEG = float(os.environ.get("PHANTOM_MAX_YAW_DEG", "5.0"))
if os.path.isfile(RING_TO_ENTRY_OFFSET_NPZ):
    with np.load(RING_TO_ENTRY_OFFSET_NPZ, allow_pickle=False) as _yaw_model:
        if "training_yaw_deg" in _yaw_model:
            PHANTOM_TRAIN_YAW_DEG = float(_yaw_model["training_yaw_deg"])



def ring_pixels_depth_mm(color_bgr, depth_mm, cx, cy, r_px, rng=None):
    """Depth (mm) of the ring's plane at the ring centre. Red (a* > RING_A_STAR_THRESH)
    pixels in the annulus RING_DEPTH_ANNULUS*r_px are fit with a RANSAC plane in
    (u, v, 1/z) -- exact for a planar ring under pinhole projection -- so near-side
    red clutter (tool, raised outer rim) and hole pixels are rejected as outliers.
    Returns (depth_mm or nan, n_inliers, n_candidates)."""
    coef, n_inl, n = ring_plane_fit(color_bgr, depth_mm, cx, cy, r_px, rng)
    if coef is None:
        return float("nan"), n_inl, n
    return float(1.0 / coef[2]), n_inl, n


def ring_plane_normal_cam(coef, cx, cy, fx, fy, ppx, ppy):
    """Unit normal (camera frame, facing the camera) of the plane
    1/z = a*(u-cx) + b*(v-cy) + c fitted by ring_plane_fit()."""
    a, b, c = coef
    n = np.array([a * fx, b * fy, a * (ppx - cx) + b * (ppy - cy) + c], np.float64)
    n /= np.linalg.norm(n)
    return -n if n[2] > 0 else n


def ring_plane_fit(color_bgr, depth_mm, cx, cy, r_px, rng=None):
    """RANSAC plane over the red ring annulus, see ring_pixels_depth_mm().
    Returns (coef [a, b, c] of 1/z = a*(u-cx) + b*(v-cy) + c, or None, n_inliers, n_candidates)."""
    h, w = depth_mm.shape
    r_in, r_out = RING_DEPTH_ANNULUS[0] * float(r_px), RING_DEPTH_ANNULUS[1] * float(r_px)
    x0, x1 = max(0, int(cx - r_out)), min(w, int(cx + r_out) + 1)
    y0, y1 = max(0, int(cy - r_out)), min(h, int(cy + r_out) + 1)
    if x1 <= x0 or y1 <= y0:
        return None, 0, 0
    a = cv2.cvtColor(color_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[:, :, 1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = depth_mm[y0:y1, x0:x1]
    rad2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = (rad2 >= r_in * r_in) & (rad2 <= r_out * r_out) & (a > RING_A_STAR_THRESH) & (d > 0)
    n = int(m.sum())
    if n < RING_DEPTH_MIN_PX:
        return None, 0, n
    u = xx[m].astype(np.float64) - cx
    v = yy[m].astype(np.float64) - cy
    z = d[m].astype(np.float64)
    M = np.stack([u, v, np.ones_like(u)], axis=1)
    iz = 1.0 / z
    rng = rng if rng is not None else np.random.RandomState(0)
    best = None
    for _ in range(200):
        idx = rng.choice(n, 3, replace=False)
        try:
            coef = np.linalg.solve(M[idx], iz[idx])
        except np.linalg.LinAlgError:
            continue
        pred = M.dot(coef)
        ok = pred > 0
        inl = ok & (np.abs(1.0 / np.where(ok, pred, 1.0) - z) < RING_DEPTH_RANSAC_MM)
        cnt = int(inl.sum())
        if best is None or cnt > best[0]:
            best = (cnt, inl)
    if best is None or best[0] < max(RING_DEPTH_MIN_PX, RING_DEPTH_MIN_INLIER_FRAC * n):
        return None, (best[0] if best else 0), n
    inl = best[1]
    coef, *_ = np.linalg.lstsq(M[inl], iz[inl], rcond=None)
    if coef[2] <= 0:
        return None, int(inl.sum()), n
    return coef, int(inl.sum()), n


def fit_line_ransac(points, thresh_mm, iters, rng):
    """Total-least-squares line fit that is ROBUST to gross outliers.
    Mirrors verify_depth_on_needle.py's fit_line_ransac() exactly (same
    algorithm) so that script's empirical verdict on YOUR camera/needle
    directly validates this code path, not a different one. See
    VIS3D_RANSAC_THRESH_MM's comment for why plain SVD is not enough here.

    Repeatedly hypothesizes a line from 2 random points, counts inliers
    (perpendicular distance < thresh_mm), keeps the hypothesis with the most
    inliers, then refits direction by SVD/TLS using ONLY that inlier set
    (standard RANSAC-then-refine). Returns None if too few points, or if the
    best hypothesis still can't attract enough inliers.
    """
    n = len(points)
    if n < 4:
        return None
    best_inliers = None
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        p0, p1 = points[i], points[j]
        d = p1 - p0
        nrm = np.linalg.norm(d)
        if nrm < 1e-6:
            continue
        d = d / nrm
        diffs = points - p0
        along = diffs.dot(d)
        perp = diffs - np.outer(along, d)
        dist = np.linalg.norm(perp, axis=1)
        inliers = dist < thresh_mm
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None:
        return None
    n_in = int(best_inliers.sum())
    if n_in < 4:
        return None
    pts_in = points[best_inliers]
    centroid = pts_in.mean(axis=0)
    X = pts_in - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction = Vt[0] / np.linalg.norm(Vt[0])
    along = X.dot(direction)
    extent_mm = float(along.max() - along.min())
    perp = X - np.outer(along, direction)
    fit_rms_mm = float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))))
    return {"direction": direction, "n_inliers": n_in, "inlier_ratio": n_in / float(n),
            "extent_mm": extent_mm, "fit_rms_mm": fit_rms_mm,
            "mask": best_inliers}  # pt.15: exposes WHICH points were inliers,
    # needed by half_split_stability_deg() to resample only the accepted set.


def half_split_stability_deg(inlier_pts, rng, n_splits=VIS3D_STABILITY_SPLITS):
    """pt.15 (2026-08-21): empirical reproducibility check for a RANSAC line
    fit -- repeatedly split the ALREADY-ACCEPTED inlier set in half at
    random, refit each half independently (plain SVD -- these points already
    passed RANSAC, no further outlier rejection needed), and return the
    WORST (max, i.e. most pessimistic) angular disagreement between the two
    halves across n_splits repeats.

    This is the key new ingredient that lets the acceptance gate drop the
    naive `inlier_ratio >= 0.5` requirement without becoming reckless: a low
    ratio just means the SAMPLE was dirty, but if the recovered inlier set
    is large enough that even a random HALF of it still nails the same
    direction every time, that direction is real, regardless of how small a
    fraction of the total candidate points it came from. Conversely, if two
    random halves of the "inlier" set disagree, the fit is not actually
    well-determined yet (e.g. the inlier count is right at the edge of
    VIS3D_MIN_INLIERS_ABS, or there's still some structured contamination
    RANSAC's single threshold didn't fully separate) and should be rejected
    even if the top-line inlier_ratio/fit_rms_mm numbers look fine in
    isolation. Returns 999.0 (always fails any sane threshold) if there are
    too few points to split meaningfully.
    """
    n = len(inlier_pts)
    if n < 8:
        return 999.0
    worst = 0.0
    idx = np.arange(n)
    for _ in range(n_splits):
        rng.shuffle(idx)
        half = n // 2
        a_idx, b_idx = idx[:half], idx[half:]

        def _fit_dir(sub_idx):
            X = inlier_pts[sub_idx] - inlier_pts[sub_idx].mean(axis=0)
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            return Vt[0]

        da, db = _fit_dir(a_idx), _fit_dir(b_idx)
        cosang = np.clip(abs(np.dot(da, db)), 0.0, 1.0)
        worst = max(worst, float(np.degrees(np.arccos(cosang))))
    return worst


def measure_shaft_direction_3d(color_bgr, depth_mm, R_he, t_he, intr, roi_center_px,
                                expected_ang_img_deg=None):
    """Independent 3D measurement of the needle-shaft's physical direction
    (BASE frame) from a single RGBD frame. See the module docstring
    (REVISED point 4) for the full method and honesty/conditioning notes.

    `expected_ang_img_deg`, if given, is the 2D image-plane angle (deg,
    mod 180) predicted by projecting the kinematically-commanded
    target_dir through the hand-eye+intrinsics model (the caller computes
    this via base_to_pixel, since it needs rcm_pt/target_dir which this
    function intentionally does not take -- keeping this function's own
    math free of any dependency on the kinematic chain it is meant to
    cross-check). It is used ONLY as a detection prior, to reject
    unrelated edges (marker rim, phantom boundary, etc.) before picking
    the "dominant" shaft line; it never enters the reported angle error.

    Returns a dict with keys:
      ok (bool), direction (3,) or None, n_points, n_inliers, inlier_ratio,
      extent_mm, fit_rms_mm, stability_deg, angular_uncertainty_deg,
      fail_reason (str or None)
    inlier_ratio is diagnostic-only (see pt.15 in the module docstring) --
    the accept/reject decision is n_inliers (absolute) + fit_rms_mm +
    extent_mm + stability_deg (half-split resampling agreement), NOT
    inlier_ratio by itself.
    NEVER raises -- callers treat any exception as "measurement unavailable".
    """
    u0, v0 = roi_center_px
    win = VIS3D_ROI_RADIUS_PX
    h, w = color_bgr.shape[:2]
    if depth_mm.shape[:2] != color_bgr.shape[:2]:
        return {"ok": False, "direction": None,
                "fail_reason": "depth/color shape mismatch %s vs %s -- frames not aligned" %
                                (depth_mm.shape[:2], color_bgr.shape[:2])}
    x0, x1 = max(0, u0 - win), min(w, u0 + win)
    y0, y1 = max(0, v0 - win), min(h, v0 + win)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return {"ok": False, "direction": None, "fail_reason": "ROI clipped off-frame"}

    roi = color_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                             minLineLength=12, maxLineGap=5)
    if lines is None or len(lines) == 0:
        return {"ok": False, "direction": None, "fail_reason": "no line segments detected in ROI"}
    segs = lines[:, 0, :]
    angs = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180.0
    lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])

    def ang_diff(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    # Stage 1 (prior filter): reject segments inconsistent with the
    # kinematically-expected 2D angle, if supplied.
    if expected_ang_img_deg is not None:
        prior_ok = np.array([ang_diff(a, expected_ang_img_deg) < VIS3D_ANGLE_PRIOR_TOL_DEG for a in angs])
        if prior_ok.any():
            segs, angs, lens = segs[prior_ok], angs[prior_ok], lens[prior_ok]
        # if nothing survives the prior, fall through and use all segments
        # (better to attempt a possibly-noisier measurement than to silently
        # report nothing just because the prior itself might be off).

    # Stage 2 (self-consistency merge): take the longest surviving segment
    # as the best single estimate of "this is the shaft", then pool in any
    # other segments close to ITS angle (more points -> better-conditioned
    # line fit) rather than trusting one segment alone.
    best_i = int(np.argmax(lens))
    theta0 = angs[best_i]
    keep = np.array([ang_diff(a, theta0) < VIS3D_ANGLE_MERGE_DEG for a in angs])
    kept_segs = segs[keep]

    pts_cam = []
    for (x1_, y1_, x2_, y2_) in kept_segs:
        seg_len = float(np.hypot(x2_ - x1_, y2_ - y1_))
        n_samp = max(2, int(seg_len / max(VIS3D_SAMPLE_STEP_PX, 1e-6)))
        for s in np.linspace(0.0, 1.0, n_samp):
            px = int(round(x1_ + s * (x2_ - x1_))) + x0
            py = int(round(y1_ + s * (y2_ - y1_))) + y0
            if px < 0 or py < 0 or px >= w or py >= h:
                continue
            z_mm = depth_min_at(depth_mm, px, py, VIS3D_DEPTH_WIN_PX)
            if not np.isfinite(z_mm) or z_mm <= 0:
                continue
            p_cam = deproject_mm(intr, px, py, z_mm / 1000.0)
            pts_cam.append(p_cam)

    if len(pts_cam) < VIS3D_MIN_POINTS:
        return {"ok": False, "direction": None,
                "fail_reason": "only %d valid 3D points (<%d needed)" % (len(pts_cam), VIS3D_MIN_POINTS)}

    pts_cam = np.array(pts_cam, float)
    pts_base = (R_he.dot(pts_cam.T)).T + t_he

    # RANSAC, not plain SVD: depth dropout on a thin/metallic shaft (multi-
    # path interference, specular reflection missing the IR return) produces
    # OUTLIER points, not just extra Gaussian noise around the true line --
    # see verify_depth_on_needle.py. A plain least-squares/SVD fit has zero
    # protection against even one such point pulling the whole direction
    # estimate off; RANSAC finds the largest self-consistent subset first.
    fit = fit_line_ransac(pts_base, VIS3D_RANSAC_THRESH_MM, VIS3D_RANSAC_ITERS, _VIS3D_RNG)
    if fit is None:
        return {"ok": False, "direction": None, "n_points": len(pts_cam),
                "fail_reason": "RANSAC line fit failed (no self-consistent subset found "
                                "among %d candidate 3D points -- depth on this shaft may be "
                                "too unreliable right now; run verify_depth_on_needle.py)" %
                                len(pts_cam)}

    direction = fit["direction"]
    extent_mm = fit["extent_mm"]
    fit_rms_mm = fit["fit_rms_mm"]
    inlier_ratio = fit["inlier_ratio"]   # pt.15: still computed + logged, diagnostic only now
    n_inliers = fit["n_inliers"]

    # pt.15 (2026-08-21) acceptance gate -- see module docstring pt.15 for
    # the full rationale for why this replaced a bare `inlier_ratio >= 0.5`
    # check. Order matters: cheapest/most-fundamental checks first.
    if extent_mm < VIS3D_MIN_EXTENT_MM:
        return {"ok": False, "direction": direction, "n_points": len(pts_cam),
                "n_inliers": n_inliers, "inlier_ratio": inlier_ratio,
                "extent_mm": extent_mm, "fit_rms_mm": fit_rms_mm,
                "fail_reason": "spatial extent %.1fmm < %.1fmm -- too short a segment to "
                                "constrain a 3D direction (ill-conditioned, same issue as "
                                "a too-small patch in phantom_normal_estimate.py)" %
                                (extent_mm, VIS3D_MIN_EXTENT_MM)}

    if n_inliers < VIS3D_MIN_INLIERS_ABS:
        return {"ok": False, "direction": direction, "n_points": len(pts_cam),
                "n_inliers": n_inliers, "inlier_ratio": inlier_ratio,
                "extent_mm": extent_mm, "fit_rms_mm": fit_rms_mm,
                "fail_reason": "only %d RANSAC inliers in absolute terms (<%d needed, ratio was "
                                "%.0f%% of %d candidates) -- too few good points to condition a "
                                "3D direction regardless of what fraction of the (heavily "
                                "specular-contaminated, see verify_depth_on_needle.py) candidate "
                                "set they represent" %
                                (n_inliers, VIS3D_MIN_INLIERS_ABS, 100 * inlier_ratio, len(pts_cam))}

    if fit_rms_mm > VIS3D_MAX_FIT_RMS_MM:
        return {"ok": False, "direction": direction, "n_points": len(pts_cam),
                "n_inliers": n_inliers, "inlier_ratio": inlier_ratio,
                "extent_mm": extent_mm, "fit_rms_mm": fit_rms_mm,
                "fail_reason": "RANSAC inliers fit RMS %.2fmm > %.2fmm -- inliers only barely "
                                "inside the %.1fmm association radius, not tightly on one line; "
                                "the RANSAC threshold accepted them but the fit is not tight "
                                "enough to trust" %
                                (fit_rms_mm, VIS3D_MAX_FIT_RMS_MM, VIS3D_RANSAC_THRESH_MM)}

    # Reproducibility check (pt.15): the real ingredient that lets us drop
    # the ratio gate safely -- does an independent random HALF of the
    # inlier set recover the same direction? If yes, the direction is real
    # no matter how small a fraction of the total sample it came from.
    stability_deg = half_split_stability_deg(pts_base[fit["mask"]], _VIS3D_STABILITY_RNG)
    if stability_deg > VIS3D_MAX_STABILITY_DEG:
        return {"ok": False, "direction": direction, "n_points": len(pts_cam),
                "n_inliers": n_inliers, "inlier_ratio": inlier_ratio,
                "extent_mm": extent_mm, "fit_rms_mm": fit_rms_mm, "stability_deg": stability_deg,
                "fail_reason": "half-split resampling disagreement %.1fdeg > %.1fdeg -- random "
                                "halves of the inlier set do not recover the same direction, so "
                                "this fit is not yet reproducible/well-determined even though it "
                                "passed the RMS/count checks" %
                                (stability_deg, VIS3D_MAX_STABILITY_DEG)}

    analytic_unc_deg = float(np.degrees(np.arctan2(fit_rms_mm, max(extent_mm / 2.0, 1e-6))))
    # Report the MORE PESSIMISTIC of the two uncertainty estimates -- a fit
    # that looks tight by RMS alone but disagrees under resampling should
    # not get to claim the smaller (falsely confident) analytic number.
    angular_uncertainty_deg = max(analytic_unc_deg, stability_deg)
    return {"ok": True, "direction": direction, "n_points": len(pts_cam),
            "n_inliers": n_inliers, "inlier_ratio": inlier_ratio, "stability_deg": stability_deg,
            "extent_mm": extent_mm, "fit_rms_mm": fit_rms_mm,
            "angular_uncertainty_deg": angular_uncertainty_deg, "fail_reason": None}


HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_TXT = (235, 235, 235)
HUD_DIM = (160, 160, 160)
HUD_OK = (100, 220, 120)
HUD_WARN = (0, 175, 255)
HUD_BAD = (70, 70, 255)
HUD_ACCENT = (255, 205, 90)
_hud = {"help": False}
_present = {"on": os.environ.get("PRESENT", "1") != "0"}
MIC_THUMB_W = {"w": 260}
# Once the RCM target is locked, the limbus / marker-ring geometry shown in
# the microscope view is frozen too (median of the frames just before lock)
# so the rings stop wandering with per-frame detection noise.
_mic_geo_freeze = {"geo": None, "buf": []}
# Demo header shows "Occluded" only after this many consecutive occluded frames
# (~4% of static frames fail mark detection individually at 2 Hz).
MIC_OCC_SHOW_N = int(os.environ.get("MIC_OCC_SHOW_N", "3"))
_mic_occ_run = {"path": None, "n": 0}
DEMO_TITLE = os.environ.get("DEMO_TITLE", "Autonomous Bi-Planar Scleral Trocar Insertion")
DEMO_SUBTITLE = os.environ.get("DEMO_SUBTITLE", "Entry point cross-verified by two independent vision systems  \u00b7  Leica microscope + RealSense D405")
# microscope inset palette (BGR): limited, desaturated
MIC_C_LIMBUS = (70, 180, 240)      # amber  -- limbus circle
MIC_C_MARKRING = (225, 190, 90)    # steel blue -- concentric marker ring
MIC_C_LOCK = (130, 215, 130)       # soft green -- RCM locked
MIC_C_SEEK = (80, 170, 245)        # orange -- RCM still settling


def hud_panel(img, x0, y0, x1, y1, alpha=0.55):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi.astype(np.float32) * (1.0 - alpha)).astype(roi.dtype)


try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
except Exception:  # pragma: no cover - falls back to OpenCV text
    _PILImage = None
_HUD_TTF = {
    False: "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    True: "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
}
_hud_font_cache = {}


def _hud_pil_font(scale, bold=False):
    if _PILImage is None:
        return None
    size = max(8, int(round(scale * 30)))
    key = (size, bool(bold))
    if key not in _hud_font_cache:
        try:
            _hud_font_cache[key] = _PILFont.truetype(_HUD_TTF[bool(bold)], size)
        except Exception:
            _hud_font_cache[key] = None
    return _hud_font_cache[key]


def hud_tw(text, scale=0.42, bold=False):
    f = _hud_pil_font(scale, bold)
    if f is None:
        return cv2.getTextSize(text, HUD_FONT, scale, 1)[0][0]
    return int(f.getsize(text)[0])


def hud_fit(text, max_w, scale, bold=False):
    if hud_tw(text, scale, bold) <= max_w:
        return text
    while text and hud_tw(text + "...", scale, bold) > max_w:
        text = text[:-1]
    return text + "..."


def hud_text(img, text, org, scale=0.42, color=HUD_TXT, thick=1, max_w=None,
             bold=False, shadow=False):
    """Draw anti-aliased sans-serif text with its baseline-left at `org`.
    Returns the drawn width in pixels."""
    if max_w:
        text = hud_fit(text, max_w, scale, bold)
    f = _hud_pil_font(scale, bold)
    if f is None:
        cv2.putText(img, text, org, HUD_FONT, scale, color, thick, cv2.LINE_AA)
        return cv2.getTextSize(text, HUD_FONT, scale, thick)[0][0]
    w = int(f.getsize(text)[0])
    size = f.size
    H, W = img.shape[:2]
    x, y = int(org[0]), int(org[1])
    x0, x1 = max(0, x - 3), min(W, x + w + 4)
    y0, y1 = max(0, y - int(size * 1.0)), min(H, y + int(size * 0.35))
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        pil = _PILImage.fromarray(np.ascontiguousarray(roi[:, :, ::-1]))
        d = _PILDraw.Draw(pil)
        if shadow:
            d.text((x - x0 + 1, y - y0 + 1), text, font=f, fill=(0, 0, 0), anchor="ls")
        d.text((x - x0, y - y0), text, font=f, fill=(int(color[2]), int(color[1]), int(color[0])),
               anchor="ls")
        roi[:] = np.asarray(pil)[:, :, ::-1]
    return w


def draw_help(view, sc=1.0):
    H_, W_ = view.shape[:2]
    groups = [
        ("SETUP", "z ring lock    n surface normal    ` mic<->D405 check + lock    y mic lock"),
        ("RUN", "h home  g approach  t ORIENT+TUNNEL  p REORIENT+PLUNGE  b retract  space STOP"),
        ("MIC CAL", "1-8 , . / ; ' - =  touch points    9 campaign touch    0 ring-offset sample"),
        ("CALIB", "SHIFT+H hand-eye sample    c origin (click tip)    SHIFT+C clear origin"),
        ("JOG", "w/s a/d r/f  xyz     k/m j/l u/o  rot     ] [  speed"),
        ("VIEW", "V demo / debug layout    Tab this help    v record"),
        ("OTHER", "x ring+offset RCM   e mic RCM (file)   SHIFT+M reset mic lock   q quit"),
    ]
    rh = int(20 * sc)
    hx0, hy0 = int(16 * sc), int(52 * sc)
    hud_panel(view, hx0, hy0, W_ - hx0, hy0 + int(26 * sc) + rh * len(groups), 0.86)
    hud_text(view, "KEYS   (Tab to close)", (hx0 + 10, hy0 + int(17 * sc)), 0.42 * sc, HUD_ACCENT, bold=True)
    for i, (g, t) in enumerate(groups):
        yy = hy0 + int(40 * sc) + rh * i
        hud_text(view, g, (hx0 + 10, yy), 0.37 * sc, HUD_ACCENT)
        hud_text(view, t, (hx0 + int(82 * sc), yy), 0.37 * sc, HUD_TXT, max_w=W_ - hx0 * 2 - int(100 * sc))


def hud_status_row(img, y, label, value, color, x=10, max_w=330):
    cv2.circle(img, (x + 4, y - 4), 3, color, -1, cv2.LINE_AA)
    hud_text(img, label, (x + 14, y), 0.38, HUD_DIM)
    hud_text(img, value, (x + 74, y), 0.40, HUD_TXT, max_w=max_w - 74)


def draw_dot(img, uv, color, label, radius=5, label_left=False, tscale=0.38):
    """Small filled marker with a thin white rim and a quiet text label."""
    if uv is None:
        return
    u, v = uv
    h, w = img.shape[:2]
    if u < -radius or v < -radius or u >= w + radius or v >= h + radius:
        return
    cv2.circle(img, (u, v), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img, (u, v), radius, color, -1, cv2.LINE_AA)
    cv2.circle(img, (u, v), radius + 1, (245, 245, 245), 1, cv2.LINE_AA)
    tw = hud_tw(label, tscale, True)
    x = (u - radius - 7 - tw) if label_left else (u + radius + 7)
    hud_text(img, label, (x, v + int(tscale * 11)), tscale, color, bold=True, shadow=True)


# Fixed superset of columns -- different actions (ORIENT vs APPROACH/INSERT/
# RETRACT) log different fields (err_deg vs err_mm, etc). A previous version
# of this function derived the CSV header from whichever action logged
# FIRST, which silently misaligned columns for every other action. Using a
# fixed fieldnames list with DictWriter(restval="", extrasaction="ignore")
# guarantees every row has the same columns in the same order regardless of
# which fields a given call happens to pass.
# 2026-10-05: the user's explicit methodology correction -- the 10(+) trial
# campaign must NOT be 10 consecutive runs from one fixed starting pose
# (that would only demonstrate repeatability from a single configuration,
# a much weaker claim); each trial must start from a genuinely DIFFERENT
# home position, to demonstrate the system generalizes across starting
# configurations, not just repeats one. _trial_state tracks this across
# 'h' presses so (a) every log row can be tagged with which trial/home it
# belongs to, for the paper's methodology section, and (b) check_home_diversity()
# can WARN (not block -- a false positive here must never stop a real demo)
# if a newly-set home is suspiciously close to one already used this
# session, catching the realistic failure mode of forgetting to actually
# move the robot between trials.
HOME_DIVERSITY_MIN_MM = float(os.environ.get("HOME_DIVERSITY_MIN_MM", "5.0"))
_trial_state = {"idx": 0, "home_history": [], "normal_fresh": False}  # home_history: list of (idx, xyz) tuples


def completed_trial_count():
    if not os.path.isdir(TRIAL_STILLS_DIR):
        return 0
    return len([f for f in os.listdir(TRIAL_STILLS_DIR) if f.startswith("trial") and f.endswith(".json")])


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)


def note_new_trial_home(home_xyz):
    """Call exactly once right when a NEW home is latched ('h' key). Bumps
    the trial index, warns (non-blocking) if this home is too close to any
    EARLIER one this session, and records it so log_precision/
    log_correspondence can auto-tag every subsequent row with
    (trial_index, home_xyz) without every individual log_precision() call
    site needing to pass it explicitly."""
    global MIC_RCM_ANGLE_DEG, INSERT_ANGLE_DEG
    _trial_state["idx"] = completed_trial_count() + 1
    _trial_state["normal_fresh"] = False
    idx = _trial_state["idx"]
    if INSERT_ANGLE_SCHEDULE:
        INSERT_ANGLE_DEG = INSERT_ANGLE_SCHEDULE[(idx - 1) % len(INSERT_ANGLE_SCHEDULE)]
        _target_dir_cache.update(dir=None, lock_t=None, azimuth_deg=None)
        print("  trial %d oblique insertion angle = %.1f deg off normal (INSERT_ANGLE_SCHEDULE)" % (
            idx, INSERT_ANGLE_DEG))
    if MIC_RCM_ANGLE_SCHEDULE:
        MIC_RCM_ANGLE_DEG = MIC_RCM_ANGLE_SCHEDULE[(idx - 1) % len(MIC_RCM_ANGLE_SCHEDULE)]
        print("  trial %d target angle = %+.1f deg (MIC_RCM_ANGLE_SCHEDULE entry %d/%d)" % (
            idx, MIC_RCM_ANGLE_DEG, (idx - 1) % len(MIC_RCM_ANGLE_SCHEDULE) + 1, len(MIC_RCM_ANGLE_SCHEDULE)))
    # The mark / circle-target trackers freeze once locked; a lock taken before this 'h'
    # (previous trial, or startup at the default angle) would otherwise be reused, which
    # silently kept trials 1-2 of 2026-10-12 at +25 deg and ignores a re-seated phantom.
    for i in (0, 1):
        _mic_mark_live["hist"][i] = []
        _mic_mark_live["smoothed"][i] = None
        _mic_mark_live["jitter_px"][i] = None
        _mic_mark_live["n"][i] = 0
        _mic_mark_live["locked"][i] = False
    _mic_mark_live["status_msg"] = "new trial: microscope mark lock reset -- re-accumulating"
    _mic_target_live.update(hist=[], smoothed=None, jitter_px=None, n=0, locked=False,
                            status_msg="new trial: circle target reset -- re-accumulating")
    home_xyz = np.asarray(home_xyz, np.float64)
    if _trial_state["home_history"]:
        dists = [float(np.linalg.norm(home_xyz - h)) for _, h in _trial_state["home_history"]]
        min_d = min(dists)
        min_i = int(np.argmin(dists))
        prev_idx = _trial_state["home_history"][min_i][0]
        if min_d < HOME_DIVERSITY_MIN_MM:
            print("  !! WARNING: trial %d's new HOME is only %.1fmm from trial %d's home "
                  "(threshold=%.1fmm) -- if you meant to start from a genuinely different "
                  "position this trial, you may have forgotten to jog the robot before "
                  "pressing 'h'. NOT blocking (this check can false-positive), but the "
                  "methodology this campaign needs ('10+ trials from different home "
                  "positions') requires this to actually be a new pose." %
                  (idx, min_d, prev_idx, HOME_DIVERSITY_MIN_MM))
        else:
            print("  trial %d HOME diversity OK: %.1fmm from the nearest previous trial's home"
                  % (idx, min_d))
    _trial_state["home_history"].append((idx, home_xyz))


PRECISION_LOG_FIELDS = ["timestamp", "action", "result", "err_deg", "err_mm",
                        "elapsed_s", "insert_angle_deg", "azimuth_deg",
                        "trial_index", "home_x_mm", "home_y_mm", "home_z_mm",
                        # independent depth-camera 3D shaft-direction cross-check
                        # (measure_shaft_direction_3d) -- NOT derived from the same
                        # kinematic chain as err_deg above, see module docstring pt.4.
                        # vis3d_inlier_ratio is the RANSAC honesty signal: low ratio
                        # means most depth samples on the shaft disagreed with any
                        # single line (specular/multi-path dropout), so the row's
                        # vis3d_err_deg (if present at all) should not be trusted.
                        "vis3d_ok", "vis3d_err_deg", "vis3d_unc_deg",
                        "vis3d_n_points", "vis3d_n_inliers", "vis3d_inlier_ratio",
                        "vis3d_extent_mm", "vis3d_rms_mm", "vis3d_stability_deg",
                        "vis3d_fail_reason",
                        # 2026-08-25 (evening): raw measured 3D shaft direction (BASE
                        # frame, sign-ambiguous -- d and -d fit equally well) PLUS the
                        # FrameEE orientation AT THE MOMENT of the check, so a
                        # systematic vis3d_err_deg (like the ~20-38deg seen tonight) can
                        # be traced back to a specific EE-frame axis hypothesis offline:
                        # NEEDLE_AXIS_LOCAL_true_ee = R(ee_q)^T @ (+-vis3d_dx/dy/dz),
                        # solved/averaged across several poses, compared to the
                        # NEEDLE_AXIS_LOCAL this run actually used (printed once at
                        # startup, constant for the whole run).
                        "vis3d_dx", "vis3d_dy", "vis3d_dz",
                        "ee_qx", "ee_qy", "ee_qz", "ee_qw"]


def log_precision(action, **fields):
    """Append one row to PRECISION_LOG. Creates the file + header on first
    use. Never raises -- a logging failure must not interrupt the demo.

    2026-10-05: auto-fills trial_index/home_{x,y,z}_mm from _trial_state/
    S.home every call -- reads S (module-level class, defined just below)
    and _trial_state (module-level dict, defined just above) directly
    rather than requiring each of this function's 9+ call sites to pass
    them explicitly, so the "which trial / which starting home" tag can
    never silently drift out of sync at some future call site that forgot
    to pass it."""
    try:
        is_new = not os.path.isfile(PRECISION_LOG)
        with open(PRECISION_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PRECISION_LOG_FIELDS,
                                extrasaction="ignore", restval="")
            if is_new:
                w.writeheader()
            row = {"timestamp": datetime.datetime.now().isoformat(), "action": action,
                   "trial_index": _trial_state["idx"]}
            if S.home is not None:
                row["home_x_mm"], row["home_y_mm"], row["home_z_mm"] = (
                    float(S.home[0]), float(S.home[1]), float(S.home[2]))
            row.update(fields)
            w.writerow(row)
    except Exception as e:
        print("  !! precision log write failed (non-fatal): %s" % e)


# 2026-10-05: dedicated log for the ONE headline ISMR number this whole
# campaign exists to produce -- "does the microscope-measured RCM point
# agree with the depth-camera-measured RCM point". Deliberately a SEPARATE,
# small, fixed-schema CSV (not just more fields bolted onto
# PRECISION_LOG_FIELDS above, and not JUST relying on ring_offset_samples.jsonl)
# so that after N>=10 presses of the new '`' key below, "mean/std/95%CI of
# discrepancy_corrected_mm" is a single pandas.read_csv(...).describe() away
# -- no joining two different files/schemas required to get the paper's
# central claim. (The '`' key ALSO still appends to ring_offset_samples.jsonl,
# see try_correspondence_check(), so none of the existing --fit/--fit-circle/
# --compare tooling loses access to this data either -- this is additive.)
CORRESPONDENCE_LOG = os.environ.get("CORRESPONDENCE_LOG", data_path("correspondence_log.csv"))
CORRESPONDENCE_LOG_FIELDS = [
    "timestamp", "trial_index", "home_x_mm", "home_y_mm", "home_z_mm",
    "mic_rcm_mode", "mic_target_mark",
    "d405_raw_x_mm", "d405_raw_y_mm", "d405_raw_z_mm",
    "d405_corrected_x_mm", "d405_corrected_y_mm", "d405_corrected_z_mm",
    "mic_x_mm", "mic_y_mm", "mic_z_mm",
    "discrepancy_raw_mm", "discrepancy_corrected_mm",
    "offset_source", "ring_jitter_mm", "ring_n",
    "mic_jitter_mm", "mic_calib_rms_mm", "mic_n",
    "occluded", "mic_still_path",
]


def log_correspondence(**fields):
    """Append one row to CORRESPONDENCE_LOG. Same never-raise, fixed-schema
    discipline as log_precision() above -- including the same 2026-10-05
    auto-fill of trial_index/home_{x,y,z}_mm from _trial_state/S.home."""
    try:
        is_new = not os.path.isfile(CORRESPONDENCE_LOG)
        with open(CORRESPONDENCE_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CORRESPONDENCE_LOG_FIELDS,
                                extrasaction="ignore", restval="")
            if is_new:
                w.writeheader()
            row = {"timestamp": datetime.datetime.now().isoformat(),
                   "trial_index": _trial_state["idx"]}
            if S.home is not None:
                row["home_x_mm"], row["home_y_mm"], row["home_z_mm"] = (
                    float(S.home[0]), float(S.home[1]), float(S.home[2]))
            row.update(fields)
            w.writerow(row)
    except Exception as e:
        print("  !! correspondence log write failed (non-fatal): %s" % e)


class S:
    home = None
    hover_goal = None
    insert_goal = None
    internal_rcm = None   # tip position at end of TUNNEL -- new pivot for PLUNGE re-orient
    tunneled = False       # gates 'p' (PLUNGE): must TUNNEL successfully first
    state = "IDLE"
    msg = ""
    rec = RECORD
    approached = False
    rcm_source = "manual"
    last_vis_check_msg = ""
    oriented = False   # True only right after a successful 't' (ORIENT); see invalidation
                        # sites below -- 'g'/APPROACH refuses to run on stale/no orientation
    warned_no_q3 = False  # pt.18 (2026-09-28): print the no-live-q3-feed warning once, not every tick


def main():
    global SPEED, ASPEED
    from runtime_session import startup_check
    startup_check()
    # Must happen before ANY rospy.Subscriber/Publisher/get_time call -- in
    # particular before the tip_inee_guard.assert_tip_inee() check below,
    # which otherwise silently times out (its Subscriber never connects
    # without an initialized node) instead of actually checking TipInEE.
    # (2026-08-24: this ordering bug was introduced when the guard call was
    # added further down, before the pre-existing rospy.init_node() call.)
    rospy.init_node("home_to_rcm_insert_v2", anonymous=True, disable_signals=True)
    # 2026-10-05: must come AFTER init_node (rospy.Subscriber requires a node)
    # and BEFORE the control loop starts -- see MIC_ROS_LIVE's docstring.
    # `global` (not a plain assignment) is deliberate: this rebinds the true
    # MODULE-level _mic_ros_live so every nested function below (which were
    # all already closing over it as a free variable from module scope)
    # observes this exact object, instead of silently creating a NEW
    # main()-local shadow that only main()'s own body could see.
    global _mic_ros_live
    if MIC_ROS_LIVE:
        _mic_ros_live = _MicRosLive(MIC_ROS_TOPIC)
        _mic_ros_live.start()
    R_he, t_he = load_handeye(NPZ)
    handeye_ok = R_he is not None

    # 2026-08-25 (later same day): model-selection block MOVED UP (was after
    # the tip_offset_pivot.npz block below) so that (a) that file's OWN
    # TipInEE guard can be SKIPPED when a better, already-guarded model
    # (rcmoffset/augmented) is active instead of hard-aborting on a stale,
    # now-superseded snapshot, and (b) NEEDLE_AXIS_LOCAL (used only by
    # ORIENT/'t') can be derived from whichever model is ACTUALLY driving
    # the tip math, not always from the legacy rigid pivot file. Root cause
    # of the abort this fixes: tip_offset_pivot.npz is a SNAPSHOT frozen at
    # whatever TipInEE was live during ITS OWN pivot_calibration.py run
    # (e.g. an earlier-today baseline of (0.21388, 0.60077, -20.16373)) --
    # a DIFFERENT, now-superseded session from the one
    # tip_offset_rcmoffset.npz's tip_inee_at_calib was corrected against via
    # handeye_vs_tip_discriminate.py. Both files can legitimately disagree;
    # only the ACTIVE model's own baseline (guarded below, in its own
    # branch) should be allowed to gate startup.
    AUG_D0, AUG_C, AUG_R_MEAN, aug_ok, aug_quality = load_tip_offset_pivot_augmented(
        TIP_OFFSET_PIVOT_AUGMENTED_NPZ)
    RCM_D_CONST, RCM_K, rcm_ok, rcm_quality = load_tip_offset_rcmoffset(
        TIP_OFFSET_RCMOFFSET_NPZ)
    use_augmented = False
    use_rcmoffset = False
    if TIP_MODEL == "rcmoffset":
        if rcm_ok:
            use_rcmoffset = True
            print("Tip offset model: RCMOFFSET (physically-derived, rcmoffset_fit.py) "
                  "d_const=%s K=%.3fmm n=%s"
                  % (RCM_D_CONST, RCM_K, rcm_quality.get("n", "?") if rcm_quality else "?"))
            if rcm_quality and "loo_rms_mm" in rcm_quality:
                print("  LOO RMS: rigid=%.3fmm -> rcmoffset=%.3fmm  (%.0f%% reduction, 1 extra "
                      "physically-grounded parameter vs augmented's 9)"
                      % (rcm_quality.get("loo_rms_rigid_mm", float("nan")), rcm_quality["loo_rms_mm"],
                         100.0 * (rcm_quality.get("loo_rms_rigid_mm", 0) - rcm_quality["loo_rms_mm"])
                         / rcm_quality.get("loo_rms_rigid_mm", 1)))
            print("  interpretation: this K is the data-fit value of robotModel.cpp's "
                  "RCM2_1.RCMoffset, which is hardcoded to 0.0 in firmware and never set "
                  "elsewhere -- see rcmoffset_fit.py docstring. NOT yet applied as an actual "
                  "firmware fix (robot-wide kinematic recalibration, higher risk); this is the "
                  "Python-side equivalent correction.")
            if rcm_quality and "tip_inee_at_calib" in rcm_quality:
                from tip_inee_guard import assert_tip_inee
                assert_tip_inee(expected=rcm_quality["tip_inee_at_calib"],
                                 context="home_to_rcm_insert_v2_precision.py start "
                                         "(must match tip_inee_at_calib in %s)" % TIP_OFFSET_RCMOFFSET_NPZ)
        else:
            print("Tip offset model: TIP_MODEL=rcmoffset requested but %s missing/incomplete -- "
                  "falling back to rigid TIP_D. Run rcmoffset_fit.py --save first."
                  % TIP_OFFSET_RCMOFFSET_NPZ)
    elif TIP_MODEL == "augmented":
        if aug_ok:
            use_augmented = True
            print("Tip offset model: AUGMENTED (orientation-dependent, pivot_residual_fit.py) "
                  "d0=%s C_frob=%.2f  n=%s"
                  % (AUG_D0, np.linalg.norm(AUG_C), aug_quality.get("n", "?") if aug_quality else "?"))
            if aug_quality and "loo_rms_augmented" in aug_quality:
                print("  loo(augmented)=%.3fmm loo(baseline)=%.3fmm (model-selection LOO on fit set)"
                      % (aug_quality["loo_rms_augmented"], aug_quality["loo_rms_baseline"]))
            if aug_quality and "fit_rms_combined_mm" in aug_quality:
                print("  fit_rms_combined=%.3fmm (training fit on all %s samples, not LOO)"
                      % (aug_quality["fit_rms_combined_mm"], aug_quality.get("n", "?")))
            print("  2026-08-25 INDEPENDENT VALIDATION: fit on n=43 pivot set, evaluated on a "
                  "SEPARATE n=1160 held-out tip_pin_check_v3_fullpose.py sweep (35.7deg spread, "
                  "never used for fitting) -- RMS-from-mean 0.321->0.118mm (-63%%), "
                  "max-from-mean 0.713->0.297mm (MARGINAL->GOOD). This npz was then refit once "
                  "more on the UNION of both datasets (lambda fixed at the value already chosen "
                  "on the n=43 set, no new grid-search) to deploy the most data-efficient model.")
            if aug_quality and "tip_inee_at_calib" in aug_quality:
                from tip_inee_guard import assert_tip_inee
                assert_tip_inee(expected=aug_quality["tip_inee_at_calib"],
                                 context="home_to_rcm_insert_v2_precision.py start "
                                         "(must match tip_inee_at_calib in %s)" % TIP_OFFSET_PIVOT_AUGMENTED_NPZ)
        else:
            print("Tip offset model: TIP_MODEL=augmented requested but %s missing/incomplete -- "
                  "falling back to rigid TIP_D. Run pivot_residual_fit.py --save first."
                  % TIP_OFFSET_PIVOT_AUGMENTED_NPZ)
    else:
        print("Tip offset model: rigid (TIP_D only) -- TIP_MODEL=%s was explicitly requested. "
              "NOTE: this baseline is known biased ~9.4mm from the true optical tip (see "
              "tip_inee_guard.py); TIP_MODEL=rcmoffset (now the default) has that correction "
              "applied and should be used unless you have a specific reason not to."
              % TIP_MODEL)

    TIP_D, tip_d_ok, tip_d_quality = load_tip_offset_pivot(TIP_OFFSET_PIVOT_NPZ)
    if tip_d_ok:
        print("Tip offset: %s  OK d=%s mm (EE-frame rigid, pivot-calibrated -- "
              "valid at any orientation)" % (TIP_OFFSET_PIVOT_NPZ, TIP_D))
        if tip_d_quality:
            q = tip_d_quality
            best_rms = q.get("loo_rms_mm", q.get("fit_rms_mm"))
            verdict = ("GOOD" if (best_rms is not None and best_rms < 0.3) else
                       "MARGINAL" if (best_rms is not None and best_rms < 0.8) else
                       "POOR" if best_rms is not None else "UNKNOWN")
            print("  pivot-cal quality: n=%s spread=%s fit_rms=%s loo_rms=%s age=%s -> %s"
                  "  -- %s" %
                  (q.get("n", "?"),
                   ("%.1fdeg" % q["spread_deg"]) if "spread_deg" in q else "?",
                   ("%.3fmm" % q["fit_rms_mm"]) if "fit_rms_mm" in q else "?",
                   ("%.3fmm" % q["loo_rms_mm"]) if "loo_rms_mm" in q else "n/a(<5 samples)",
                   ("%.1fhr" % q["age_hr"]) if "age_hr" in q else "?",
                   verdict,
                   ("this file is INACTIVE for the tip position (TIP_MODEL=%s is driving both "
                    "that and NEEDLE_AXIS_LOCAL below); shown for reference only" % TIP_MODEL)
                   if (use_rcmoffset or use_augmented) else
                   "THIS OFFSET SETS NEEDLE_AXIS_LOCAL BELOW, so a bad number here biases BOTH "
                   "the achieved 3D angle AND the RCM-insertion point, and the software's own "
                   "error readout cannot detect that bias (see README 补充5)"))
            if verdict in ("MARGINAL", "POOR", "UNKNOWN") and not (use_rcmoffset or use_augmented):
                print("  !! re-run pivot_calibration.py (wide orientation sweep, >=8-10 samples) "
                      "before trusting the achieved insertion angle/point in this demo.")
            if "tip_inee_at_calib" in q:
                if use_rcmoffset or use_augmented:
                    # 2026-08-25: do NOT hard-abort on THIS FILE's own frozen
                    # baseline when a different, already-guarded model
                    # (rcmoffset/augmented, guarded above) is the one
                    # actually driving the tip math -- see the comment on
                    # the model-selection block above for why the two files
                    # are allowed to disagree. Informational only.
                    try:
                        from tip_inee_guard import read_live_tip_inee
                        live_now = read_live_tip_inee(timeout_s=3.0)
                        err = np.abs(live_now - np.asarray(q["tip_inee_at_calib"], float))
                        if np.any(err > 0.5):
                            print("  (FYI, not fatal: live TipInEE %s differs from THIS FILE's "
                                  "own frozen pivot-session baseline %s by %s mm -- expected/"
                                  "harmless since TIP_MODEL=%s is what's actually active, not "
                                  "this file)"
                                  % (np.round(live_now, 3), np.round(q["tip_inee_at_calib"], 3),
                                     np.round(err, 3), TIP_MODEL))
                    except Exception as e:
                        print("  (could not read live TipInEE for the informational check: %s)" % e)
                else:
                    # 2026-08-24: this is the actual root-cause guard for the
                    # "tip dot sometimes accurate, sometimes not" saga -- TipInEE
                    # has silently drifted across sessions multiple times, each
                    # time silently invalidating TIP_D above. Hard-abort rather
                    # than warn-and-continue, since a stale TIP_D biases BOTH the
                    # tip position AND (via NEEDLE_AXIS_LOCAL below) the achieved
                    # insertion angle in a way this script cannot self-detect.
                    # Only reached when this file IS the active model
                    # (TIP_MODEL=rigid, or rcmoffset/augmented unavailable).
                    from tip_inee_guard import assert_tip_inee
                    assert_tip_inee(expected=q["tip_inee_at_calib"],
                                     context="home_to_rcm_insert_v2_precision.py start "
                                             "(must match tip_inee_at_calib in %s)" % TIP_OFFSET_PIVOT_NPZ)
            else:
                print("  !! tip_offset_pivot.npz has no tip_inee_at_calib recorded (pre-2026-08-24 "
                      "file) -- cannot verify live TipInEE matches. Re-run pivot_calibration.py "
                      "to get this safety check.")
    else:
        TIP_W, tip_w_ok = load_tip_offset(TIP_OFFSET_NPZ)
        print("Tip offset: %s -- %s (falling back to constant-BASE-frame model, only"
              " valid near the orientation it was collected at)" %
              (TIP_OFFSET_NPZ,
               ("OK w=%s mm" % TIP_W) if tip_w_ok else "MISSING (assuming FrameEE == tip)"))

    # 2026-08-25 BUG FOUND: this used to be `NEEDLE_AXIS_LOCAL = TIP_D /
    # norm(TIP_D)` -- but per pivot_calibration.py's own docstring, TIP_D is
    # a RESIDUAL correction on top of whatever TipInEE the firmware already
    # has baked in (e.g. (0,0,-22)), NOT the full EE-frame offset to the
    # tip. Normalizing the bare residual alone gives a essentially made-up
    # direction -- with today's TIP_D=[0.21,0.60,1.84] (small, +Z-dominant)
    # vs. the correct full offset [0.21,0.60,-20.16] (-Z-dominant), the two
    # axes are 159deg apart (measured 2026-08-25). auto_orient_about_rcm()
    # rotates the tool until NEEDLE_AXIS_LOCAL (in base frame) matches the
    # drawn target/"red line" direction -- so with the buggy axis, the loop
    # converges "successfully" while the REAL needle shaft ends up ~159deg
    # off from the red line. This is why ORIENT stopped matching the red
    # line today even though nothing about ORIENT's control law changed:
    # TIP_D's sign/magnitude just happened to change enough between pivot
    # calibration sessions to flip which direction the (buggy) normalize
    # picked. Fix: use the FULL offset (firmware TipInEE-at-calibration-time
    # + TIP_D), matching what actually IS the EE-frame-to-tip vector.
    #
    # 2026-08-25 (later): when a corrected model (rcmoffset/augmented) is
    # ACTIVE, use ITS OWN tip_inee_at_calib as the axis source instead of
    # the (possibly stale/superseded) tip_offset_pivot.npz snapshot above --
    # see the guard comments above for why the two files can legitimately
    # disagree. tip_inee_at_calib for these models already IS the full
    # EE-origin-to-tip vector (not a bare residual), so no TIP_D-style
    # addition is needed; the small K*(u(q4)-ubar) shape term is at most a
    # few mm of perpendicular wobble on a ~13-22mm vector and is not worth
    # the extra complexity of reading a live joint angle just to pick an
    # axis REFERENCE direction for ORIENT.
    if use_rcmoffset and rcm_quality and "tip_inee_at_calib" in rcm_quality:
        _needle_full = np.asarray(rcm_quality["tip_inee_at_calib"], float)
        NEEDLE_AXIS_LOCAL = _needle_full / np.linalg.norm(_needle_full)
        print("  NEEDLE_AXIS_LOCAL (EE frame) = %s  (from RCMOFFSET tip_inee_at_calib=%s)" %
              (np.round(NEEDLE_AXIS_LOCAL, 4), np.round(_needle_full, 3)))
    elif use_augmented and aug_quality and "tip_inee_at_calib" in aug_quality:
        _needle_full = np.asarray(aug_quality["tip_inee_at_calib"], float)
        NEEDLE_AXIS_LOCAL = _needle_full / np.linalg.norm(_needle_full)
        print("  NEEDLE_AXIS_LOCAL (EE frame) = %s  (from AUGMENTED tip_inee_at_calib=%s)" %
              (np.round(NEEDLE_AXIS_LOCAL, 4), np.round(_needle_full, 3)))
    elif tip_d_ok and np.linalg.norm(TIP_D) > 1e-6:
        _tip_inee_base = None
        if tip_d_quality is not None:
            _tip_inee_base = tip_d_quality.get("tip_inee_at_calib")
        if _tip_inee_base is None:
            _tip_inee_base = np.zeros(3)
            print("  !! tip_offset_pivot.npz has no tip_inee_at_calib -- NEEDLE_AXIS_LOCAL is "
                  "using TIP_D alone (residual-only), which is WRONG whenever the firmware "
                  "TipInEE baseline is nonzero (e.g. the usual (0,0,-22)). Re-run "
                  "pivot_calibration.py to regenerate with this field.")
        _needle_full = np.asarray(_tip_inee_base, float) + TIP_D
        NEEDLE_AXIS_LOCAL = _needle_full / np.linalg.norm(_needle_full)
        print("  NEEDLE_AXIS_LOCAL (EE frame) = %s  (from TipInEE_at_calib=%s + TIP_D=%s)" %
              (np.round(NEEDLE_AXIS_LOCAL, 4), np.asarray(_tip_inee_base, float), np.round(TIP_D, 3)))
    else:
        NEEDLE_AXIS_LOCAL = np.array([0.0, 0.0, -1.0])

    # Current independent axis only; no historical override or position-derived fallback.
    from runtime_session import load_axis
    NEEDLE_AXIS_LOCAL = load_axis(required=os.environ.get("ISMR_MODE") == "insertion")

    PHANTOM_NORMAL = load_phantom_normal(NORMAL_NPZ)
    if PHANTOM_NORMAL is not None:
        print("Phantom normal: %s  OK n=[%.3f %.3f %.3f] (fit rms=%.2fmm over %d pts) -- "
              "target insertion angle = %.1fdeg off normal, azimuth=%s" %
              (NORMAL_NPZ, PHANTOM_NORMAL["normal"][0], PHANTOM_NORMAL["normal"][1],
               PHANTOM_NORMAL["normal"][2], PHANTOM_NORMAL["rms"], PHANTOM_NORMAL["n_points"],
               INSERT_ANGLE_DEG,
               ("%.1fdeg (fixed, two-axis frame -- pt.16)" % INSERT_AZIMUTH_DEG)
               if INSERT_AZIMUTH_DEG is not None
               else "auto (LEGACY single-axis mode, INSERT_AZIMUTH_DEG=legacy set explicitly "
                    "-- NOT reproducible, see pt.16)"))
    else:
        print("Phantom normal: %s MISSING -- run phantom_normal_estimate.py first for angled "
              "('t') insertion; g/i/b still work at whatever angle the tool is already at." %
              NORMAL_NPZ)

    def tip_offset_vec():
        if _ee["q"] is None:
            return np.zeros(3)
        R = quat_to_R(_ee["q"])
        if use_rcmoffset:
            q4 = np.arctan2(-R[1, 2], R[1, 1])
            return R.dot(RCM_D_CONST + RCM_K * np.array([0.0, np.sin(q4), np.cos(q4)]))
        if use_augmented:
            r = Rot.from_matrix(AUG_R_MEAN.T.dot(R)).as_rotvec()
            return R.dot(AUG_D0 + AUG_C.dot(r))
        if tip_d_ok:
            return R.dot(TIP_D)
        return TIP_W

    def cur_tip():
        return None if _ee["p"] is None else _ee["p"] + tip_offset_vec()

    pub = rospy.Publisher(VEL_TOPIC, Vector3, queue_size=1)
    pub_a = rospy.Publisher(ANG_TOPIC, Vector3, queue_size=1)
    # 2026-10-06: found while chasing "tip dot can't track in real time" --
    # these three Subscriber() calls had NO queue_size at all, which rospy
    # defaults to UNBOUNDED. FrameEE publishes at ~193Hz (confirmed live via
    # `rostopic hz`); this script's main thread does per-frame cv2/numpy
    # work (image decode, ring/aruco detection, drawing) that holds the GIL
    # for stretches long enough that the subscriber callback thread cannot
    # always keep up at 193Hz. With an unbounded queue, a slow consumer does
    # NOT drop old messages -- it just processes them in strict FIFO order
    # with ever-GROWING latency, i.e. _ee["p"]/cur_tip() silently falls
    # further and further behind the robot's true current pose the longer
    # the process runs, exactly matching the reported symptom (persisted
    # through the cv2.setNumThreads(1) fix, which only reduced *contention*
    # overhead -- it never touched this separate unbounded-backlog bug).
    # Fix: queue_size=1 makes rospy drop anything older than the newest
    # unprocessed message instead of queuing it -- for a live pose feed we
    # only ever want the LATEST value anyway, never catch-up-on-history.
    # tcp_nodelay=True additionally disables TCP/Nagle-style batching on the
    # wire for these three low-latency control-feedback topics (mirrors the
    # mic image subscriber's own queue_size=1 pattern at _MicRosLive._sub
    # above, which never had this bug).
    rospy.Subscriber(FRAMEEE_TOPIC, Transform, _cb_ee, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber(RCMPOINT_TOPIC, Vector3, _cb_rcm, queue_size=1, tcp_nodelay=True)
    rospy.Subscriber(JOINTPOS_TOPIC, Float64MultiArray, _cb_jointpos, queue_size=1, tcp_nodelay=True)

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    configure_d405_manual_exposure(profile)  # pt.23 -- must run before warm-up, see docstring
    intr = load_best_intrinsics(profile)
    align = rs.align(rs.stream.color)  # aligns depth pixel coords to the color frame
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    for _ in range(15):
        pipe.wait_for_frames()

    time.sleep(0.5)
    if cur_tip() is not None:
        S.home = cur_tip().copy()
        print("HOME latched at startup (tip-corrected): [%.2f %.2f %.2f]" % tuple(S.home))

    os.makedirs(RECORD_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path = os.path.join(RECORD_DIR, "home_rcm_v2_%s.mp4" % stamp)
    REC_FPS = 20.0
    vw = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), REC_FPS, (1280, 720))
    # The UI loop runs at only ~2 Hz, so frames are written against the wall clock
    # (each view repeated until the file catches up) to keep playback at 1x.
    # A gap > REC_RESUME_GAP_S (recording toggled off/on) is skipped, not filled.
    REC_RESUME_GAP_S = 5.0
    _rec_clock = {"t0": None, "n": 0, "last": 0.0}
    if RECORD:
        print("recording -> %s" % vid_path)

    def zero():
        pub.publish(Vector3(0, 0, 0)); pub_a.publish(Vector3(0, 0, 0))

    def rcm_ok():
        if S.rcm_source == "vision":
            return _vision_rcm["p"] is not None
        return _rcm["p"] is not None and np.linalg.norm(_rcm["p"]) > 1e-3

    def cur_rcm():
        if S.rcm_source == "vision":
            return _vision_rcm["p"]
        if _rcm["p"] is None:
            return None
        return _rcm["p"] + tip_offset_vec()

    def update_live_ring_detection(color_bgr, depth_frame):
        """Runs detect_ring() on the CURRENT frame from this script's own
        camera pipe and folds the result into the SAME kind of rolling
        smoothing/jitter buffer auto_target_vision.py used, but in-process
        -- see module docstring pt.6. Called every grab_show() tick so the
        lock status is always fresh; 'z' just reads whether _ring_live is
        currently locked, it does not itself trigger a one-shot detection.
        NEVER raises -- a bad frame just updates the status message.
        """
        try:
            cx, cy, r_px, n_px, fail_reason = detect_ring(color_bgr, _ring_live["last_center"])
        except Exception as e:
            _ring_live["det"] = False
            _ring_live["status_msg"] = "ring detect error (non-fatal): %s" % e
            return
        _ring_live["det"] = cx is not None
        _ring_live["cx"], _ring_live["cy"], _ring_live["r_px"] = cx, cy, r_px
        if cx is None:
            _ring_live["status_msg"] = "RING NOT VISIBLE (%s)" % (fail_reason or "?")
            return
        _ring_live["last_center"] = (cx, cy)

        if not handeye_ok:
            _ring_live["status_msg"] = "ring seen (r=%.0fpx) but hand-eye missing" % r_px
            return
        depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float64) * depth_scale * 1000.0
        z_mm = float("nan")
        _ring_live["last_color"], _ring_live["last_depth_mm"] = color_bgr, depth_mm
        if RING_DEPTH_MODE == "ring":
            z_mm, _n_inl, _n_cand = ring_pixels_depth_mm(color_bgr, depth_mm, cx, cy, r_px)
            _ring_live["depth_src"] = "ring plane %d/%d px" % (_n_inl, _n_cand)
        if not np.isfinite(z_mm):
            z_mm = depth_min_at(depth_mm, int(round(cx)), int(round(cy)), RING_DEPTH_WIN_PX)
            _ring_live["depth_src"] = "center min"
        if not np.isfinite(z_mm) or z_mm <= 0:
            _ring_live["status_msg"] = "ring seen (r=%.0fpx) but no valid depth at center" % r_px
            return
        cam_mm = deproject_mm(intr, cx, cy, z_mm / 1000.0)  # deproject_mm wants depth in METERS
        base_pt = R_he.dot(cam_mm) + t_he

        hd = _ring_live["hist_depth"]
        hd.append(base_pt)
        if len(hd) > RING_SMOOTH_N:
            del hd[0]

        if RING_DIAM_MM > 0 and r_px > 2:
            z_size_mm = (RING_DIAM_MM * intr.fx) / (2.0 * r_px)
            cam_mm2 = deproject_mm(intr, cx, cy, z_size_mm / 1000.0)
            base_pt2 = R_he.dot(cam_mm2) + t_he
            hs = _ring_live["hist_size"]
            hs.append(base_pt2)
            if len(hs) > RING_SMOOTH_N:
                del hs[0]

        n_d = len(hd)
        if n_d < 3:
            _ring_live["status_msg"] = "ring seen, collecting depth samples (%d/3)" % n_d
            return
        arr = np.array(hd)
        smoothed_d = arr.mean(axis=0)
        jitter_d = float(arr.std(axis=0).max())
        _ring_live["smoothed"] = smoothed_d
        _ring_live["jitter"] = jitter_d
        _ring_live["n"] = n_d

        locked = (n_d >= RING_SMOOTH_N) and (jitter_d < RING_MAX_JITTER_MM)
        _ring_live["locked"] = locked
        if locked:
            diff_mm, agree = 0.0, True
            hs = _ring_live["hist_size"]
            if RING_DIAM_MM > 0 and len(hs) >= 3:
                smoothed_s = np.array(hs).mean(axis=0)
                diff_mm = float(np.linalg.norm(smoothed_d - smoothed_s))
                agree = diff_mm < RING_AGREE_THRESHOLD_MM
            try:
                save_target_atomic(AUTO_TARGET_NPZ, smoothed_d, smoothed_d, diff_mm, agree, n_d, jitter_d)
            except Exception as e:
                print("  !! auto_target.npz artifact write failed (non-fatal, live lock unaffected): %s" % e)
            _ring_live["status_msg"] = "RING LOCKED  jitter=%.2fmm n=%d  depth=%s" % (
                jitter_d, n_d, _ring_live.get("depth_src"))
        else:
            _ring_live["status_msg"] = "ring collecting %d/%d  jitter=%.2fmm" % (n_d, RING_SMOOTH_N, jitter_d)

    def try_lock_vision_rcm():
        """'z': accept the LIVE in-process ring-detection lock (see
        update_live_ring_detection(), called every frame) -- no longer reads
        auto_target.npz from a separate process. See module docstring pt.6
        for why the old file-handoff design was fragile."""
        if not _ring_live["locked"]:
            S.msg = _ring_live["status_msg"] or "ring not locked yet"
            print("  z: not locked yet -- %s" % S.msg)
            return False
        d_p = _ring_live["smoothed"].copy()
        _vision_rcm["p"] = d_p; _vision_rcm["t"] = time.time()
        _vision_rcm["n"] = _ring_live["n"]; _vision_rcm["jitter"] = _ring_live["jitter"]
        S.rcm_source = "vision"
        S.msg = "VISION RCM locked [%.1f %.1f %.1f]mm jitter=%.2fmm (live, in-process)" % (
            d_p[0], d_p[1], d_p[2], _ring_live["jitter"])
        print("  " + S.msg)
        return True

    def try_lock_vision_rcm_with_offset():
        """'x': same live D405 ring lock as 'z', PLUS a fixed base-frame
        correction (learned once via mic_extrinsic_calib.py --ring-offset,
        see RING_TO_ENTRY_OFFSET_NPZ) that shifts it from "limbus center"
        (what detect_ring() actually finds on this phantom -- it looks for
        a red/maroon annulus, and the limbus is red/maroon; the entry marks
        are small dark ink dots ~3-4mm posterior to it, nowhere near the
        limbus centroid, see chat write-up 2026-09-21) toward "the true
        ink-marked entry point". This is D405-ONLY (no microscope photo
        needed at the moment of use) so it is fine for APPROACH/HOVER, but
        it is only as good as (a) the offset-learning residual and (b) the
        phantom not having moved since that offset was learned -- it is
        NOT a substitute for 'y'/'e' right before the actual insertion
        move, which re-measures from a fresh microscope photo instead of
        trusting an old learned constant.

        2026-10-05: in MIC_RCM_MODE=="circle", the correction below is
        instead the phantom-pose-invariant, tangent-plane-LOCAL-frame
        version (radius_local_mm/theta_local_deg, see
        mic_extrinsic_calib.tangent_frame_from_normal() +
        mic_ring_offset_verify.py --fit-circle) -- it reconstructs the
        target from the CURRENT live ring lock + the CURRENT live
        PHANTOM_NORMAL (updated by 'n'), so it should remain correct even
        if the phantom has moved/rotated since the offset was learned, AS
        LONG AS 'n' has been freshly re-pressed on the CURRENT phantom
        pose. It is NOT free of that requirement -- a stale PHANTOM_NORMAL
        from session start is just as wrong as a stale fixed-vector offset
        would have been; this is a different dependency, not zero
        dependency."""
        if not _ring_live["locked"]:
            S.msg = _ring_live["status_msg"] or "ring not locked yet"
            print("  x: not locked yet -- %s" % S.msg)
            return False

        if MIC_RCM_MODE == "circle":
            if PHANTOM_NORMAL is None:
                S.msg = "no phantom normal -- press 'n' first (needed to build the local tangent frame)"
                return False
            if MIC_RCM_RADIUS_LOCAL_MM is not None:
                # 2026-10-05: explicit clinical/protocol radius overrides
                # EITHER mark's empirical npz fit -- see MIC_RCM_RADIUS_LOCAL_MM
                # declaration above for why (mark0 vs mark1 disagreed by ~5mm
                # today). No npz dependency at all in this branch.
                radius_local = MIC_RCM_RADIUS_LOCAL_MM
                loo_rms = -1.0  # not applicable -- not fit from data
                radius_src_label = "CLINICAL-FIXED"
            else:
                if not os.path.isfile(RING_TO_ENTRY_OFFSET_CIRCLE_NPZ):
                    S.msg = ("no %s -- run mic_ring_offset_verify.py --fit-circle first (needs >=5 "
                              "'0'-key samples logged AFTER pressing 'n', i.e. with phantom_normal_base "
                              "set), or set MIC_RCM_RADIUS_LOCAL_MM to use a fixed clinical radius "
                              "instead" % RING_TO_ENTRY_OFFSET_CIRCLE_NPZ)
                    return False
                try:
                    off_d = np.load(RING_TO_ENTRY_OFFSET_CIRCLE_NPZ)
                    radius_local = float(off_d["radius_local_mm_mark%d" % MIC_TARGET_MARK])
                    loo_rms = float(off_d.get("loo_rms_mm_mark%d" % MIC_TARGET_MARK, -1.0))
                except KeyError:
                    S.msg = ("%s has no local offset for mark%d -- re-run --fit-circle, or set "
                              "MIC_RCM_RADIUS_LOCAL_MM to use a fixed clinical radius instead" %
                              (RING_TO_ENTRY_OFFSET_CIRCLE_NPZ, MIC_TARGET_MARK))
                    return False
                radius_src_label = ("npz-fit-mark%d" % MIC_TARGET_MARK)
            sys.path.insert(0, MIC_AUTONOMY_DIR)
            import mic_extrinsic_calib as _mec
            normal = PHANTOM_NORMAL["normal"] / np.linalg.norm(PHANTOM_NORMAL["normal"])
            u, v = _mec.tangent_frame_from_normal(normal)
            angle = MIC_RCM_ANGLE_DEG
            d_p = (_ring_live["smoothed"].copy()
                   + radius_local * np.cos(np.radians(angle)) * u
                   + radius_local * np.sin(np.radians(angle)) * v)
            _vision_rcm["p"] = d_p; _vision_rcm["t"] = time.time()
            _vision_rcm["n"] = _ring_live["n"]; _vision_rcm["jitter"] = _ring_live["jitter"]
            S.rcm_source = "vision"
            S.msg = ("VISION RCM (ring+local-circle-offset) locked @%.0fdeg [%.1f %.1f %.1f]mm  "
                      "(ring jitter=%.2fmm, radius_local=%.1fmm [%s], loo_rms=%s) -- approach/hover "
                      "OK, still use 'y'/'e' before 'i'!" % (
                          angle, d_p[0], d_p[1], d_p[2], _ring_live["jitter"], radius_local,
                          radius_src_label,
                          ("%.3fmm" % loo_rms) if loo_rms >= 0 else "N/A"))
            print("  " + S.msg)
            return True

        if not os.path.isfile(RING_TO_ENTRY_OFFSET_NPZ):
            S.msg = ("no %s -- run mic_extrinsic_calib.py --ring-offset first (needs a fresh 'z' "
                      "lock + one microscope still at the same phantom pose)" % RING_TO_ENTRY_OFFSET_NPZ)
            return False
        try:
            off_d = np.load(RING_TO_ENTRY_OFFSET_NPZ)
            offset = np.array(off_d["offset_mark%d" % MIC_TARGET_MARK], np.float64)
        except KeyError:
            S.msg = ("%s has no offset for mark%d -- re-run --ring-offset on a still where that "
                      "mark is visible" % (RING_TO_ENTRY_OFFSET_NPZ, MIC_TARGET_MARK))
            return False
        d_p = _ring_live["smoothed"].copy() + offset
        _vision_rcm["p"] = d_p; _vision_rcm["t"] = time.time()
        _vision_rcm["n"] = _ring_live["n"]; _vision_rcm["jitter"] = _ring_live["jitter"]
        S.rcm_source = "vision"
        S.msg = ("VISION RCM (ring+learned-offset) locked mark%d [%.1f %.1f %.1f]mm  "
                  "(ring jitter=%.2fmm, offset=[%.1f %.1f %.1f]mm) -- approach/hover OK, "
                  "still use 'y'/'e' before 'i'!" % (
                      MIC_TARGET_MARK, d_p[0], d_p[1], d_p[2], _ring_live["jitter"],
                      offset[0], offset[1], offset[2]))
        print("  " + S.msg)
        return True

    def try_lock_microscope_rcm():
        """'e': load an entry-point estimate computed OFFLINE from a Leica
        microscope still (mic_extrinsic_calib.py --still --mark {0,1}, see
        /home/eyebrp/Matthew/microscope_autonomy/) and lock it as the RCM
        point exactly the way 'z' locks the live D405 ring detection --
        both just populate _vision_rcm + S.rcm_source="vision", so every
        downstream consumer (rcm_ok(), cur_rcm(), 'g'/APPROACH, 't'/ORIENT)
        works unmodified regardless of which vision source produced the
        point. This does NOT read the microscope live and does NOT verify
        the number is fresh -- it only reads whatever mic_entry_point.json
        currently contains, so ALWAYS check the printed age/source-image
        before trusting it, and re-run mic_extrinsic_calib.py --still after
        every new capture (a stale file here is worse than no file)."""
        if not os.path.isfile(MIC_ENTRY_POINT_JSON):
            S.msg = "no %s -- run mic_extrinsic_calib.py --still first" % MIC_ENTRY_POINT_JSON
            return False
        try:
            d = json.load(open(MIC_ENTRY_POINT_JSON))
        except Exception as e:
            S.msg = "could not read %s: %s" % (MIC_ENTRY_POINT_JSON, e)
            return False
        age_s = time.time() - d.get("timestamp", 0)
        p = np.array([d["X"], d["Y"], d["Z"]], np.float64)
        rms = d.get("calib_rms_mm", None)
        _vision_rcm["p"] = p; _vision_rcm["t"] = time.time()
        _vision_rcm["n"] = 1
        _vision_rcm["jitter"] = rms if rms is not None else float("nan")
        S.rcm_source = "vision"
        S.msg = ("MICROSCOPE RCM locked mark%s [%.2f %.2f %.2f]mm  calib_rms~%.3fmm  "
                 "computed %.0fs ago from %s%s -- VERIFY before 'g'/'t'!" % (
                     d.get("mark_index", "?"), p[0], p[1], p[2],
                     rms if rms is not None else float("nan"), age_s,
                     os.path.basename(d.get("source_image", "?")),
                     "  [FLAGGED OCCLUDED AT CAPTURE]" if d.get("occluded") else ""))
        print("  " + S.msg)
        return True

    # 2026-09-21: passive preview cache for the PiP panel in draw() below --
    # separate from try_lock_microscope_rcm_live() on purpose: rendering the
    # preview every frame should NOT require pressing 'y', so you can watch
    # the recognition track a newly-dropped photo live and judge whether it
    # looks right BEFORE committing to it with 'y'. Only re-runs mic_perception
    # when the newest file's mtime actually changes (cheap: os.listdir + one
    # stat per frame; the ~10-30ms detection itself only reruns on a new file).
    _mic_prev = {"path": None, "mtime": None, "thumb": None, "rec": None,
                 "dir_scan_t": 0.0}
    # 2026-10-06: throttle for the os.listdir()+max(key=getmtime) scan below --
    # see that code's comment for why. Without this, cost grows unboundedly
    # with session length (confirmed live: 14563 accumulated files in
    # MIC_LIVE_STILLS_DIR made this scan alone the dominant per-frame cost,
    # independent of and worse than MIC_ROS_LIVE's own processing -- "tip dot
    # lags" persisted even with MIC_ROS_LIVE=0 because THIS path runs on
    # every single disk-polling-fallback frame). 5Hz is plenty for a source
    # that only writes a new still every ~0.5s anyway.
    MIC_DIR_SCAN_INTERVAL_S = 0.2
    # 2026-10-06: see the matching comment at this function's locate() call
    # site below -- wall-clock floor between actual mic_perception.locate()
    # calls, independent of the file-change check.
    # 2026-10-06: bumped default from 0.5 -> 1.5 after a `py-spy record`
    # aggregate profile (8s/891 samples) attributed 54.3% of TOTAL
    # MainThread wall-clock time to get_mic_preview() (31.7% inside
    # find_plug_circle()'s single cv2.HoughCircles call alone) -- found to
    # be running on a FULL 1920x1080 microscope still (confirmed via
    # cv2.imread().shape on a live file), ~6.75x the pixel count a smaller
    # test image would have, which plausibly explains why this costs far
    # more than mic_perception.py's own "~10-30ms/frame" docstring claims.
    # Properly fixing find_plug_circle() itself (e.g. downscaling before
    # Hough + rescaling results) would require touching PLUG_MIN_R/MAX_R/
    # EXPECTED_R's calibrated pixel-radius constants in a SHARED module
    # also used by other calibration tools -- too risky to get exactly
    # right under today's time pressure, when today's actual priority
    # (redoing the microscope extrinsic calibration) depends on this exact
    # detector being correct. Throttling how OFTEN it runs is zero-risk to
    # correctness (same cached result, just updated less often) and is the
    # single highest-leverage lever available for the reported "tip dot"
    # responsiveness complaint, since mic preview freshness was never the
    # thing actually being asked about. Lower this back down (e.g. =0.5)
    # if/when actively doing microscope-preview-dependent work and the main
    # loop's responsiveness is not the current bottleneck of concern.
    # 2026-10-06 (pt.2): back down to 0.5 (its pre-throttle-bump value) now
    # that the actual locate() call runs on a background thread (below) and
    # no longer blocks grab_show() at all -- this now only controls mic
    # PiP-panel freshness, not main-loop responsiveness, so there is no
    # reason to keep it artificially high anymore.
    MIC_PROCESS_MIN_INTERVAL_S = float(os.environ.get("MIC_PROCESS_MIN_INTERVAL_S", "0.5"))
    # 2026-10-06 (pt.2, same lag investigation): throttling alone only
    # reduces how OFTEN find_plug_circle()'s slow Hough call runs -- it does
    # NOT shorten any single call, so the tip overlay still visibly hitches
    # once per MIC_PROCESS_MIN_INTERVAL_S (reported: "还是无法实时追踪" even
    # after the 0.5->1.5s throttle bump). The actual architectural fix: run
    # cv2.imread()+mic_perception.locate() in a background worker thread so
    # grab_show()'s render/control loop NEVER blocks on them at all,
    # regardless of how long any single call takes. 1 worker (not a pool)
    # is deliberate -- never want two locate() calls fighting over the
    # (intentionally single-threaded, see cv2.setNumThreads(1) above) CPU
    # at once; a 2nd due tick while one is still in flight just keeps using
    # the last cached thumb, exactly like the old synchronous throttle did
    # while "not yet due". Zero risk to detection accuracy -- same
    # mic_perception.py code, same resolution, same constants, only WHERE
    # it runs changes.
    _mic_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mic_locate")
    _mic_future = [None]

    def _mic_bg_load_and_locate(path, target_angle_deg, target_radius_px):
        sys.path.insert(0, MIC_AUTONOMY_DIR)
        import mic_perception as _mp
        img = cv2.imread(path)
        if img is None:
            raise ValueError("cv2.imread returned None for %s" % path)
        rec = _mp.locate(img, target_angle_deg=target_angle_deg, target_radius_px=target_radius_px)
        return img, rec

    def update_mic_mark_live(rec):
        """Folds this frame's mark0/mark1 centroids into _mic_mark_live's
        rolling buffer, exactly mirroring update_live_ring_detection()'s
        pattern for the D405 ring. Sticky through occlusion (a bad/occluded
        frame just leaves the previous smoothed/locked state untouched,
        same as the ring path does on a failed detection) -- one bad frame
        should not throw away several seconds of good agreement, and it
        should not silently be treated as a fresh, trustworthy sample either.
        """
        if not rec["ok"] or rec["occluded"]:
            _mic_mark_live["status_msg"] = "mic marks: frame occluded/unreliable, holding last lock"
            return
        any_collecting = False
        for i in (0, 1):
            if _mic_mark_live["locked"][i]:
                # 2026-09-28 (pt.2, real bug): once a mark is LOCKED, FREEZE
                # it -- do NOT fold in any further samples. Reported symptom:
                # "Locked之后有时候状态还在变" (state keeps changing even
                # after LOCKED). Root cause: this loop kept sliding the
                # MIC_MARK_SMOOTH_N-sample window and recomputing `smoothed`
                # on EVERY frame forever, locked or not -- so the on-screen
                # dot, the "RCM LOCKED" mm readout, and whatever 'y' actually
                # latches all kept drifting by a fraction of a px on every
                # new frame even long after first reaching LOCKED. "Locked"
                # is supposed to mean "this is the final answer" -- hist/
                # smoothed/jitter_px are left exactly as they were at the
                # instant locked first became True; no reset key exists yet
                # to intentionally re-lock a mark after physically moving
                # the phantom/scope (would need a fresh process restart, or
                # a new explicit reset action, if that's ever needed).
                continue
            mcx, mcy = rec.get("mark%d_cx" % i), rec.get("mark%d_cy" % i)
            if mcx is None:
                continue
            hist = _mic_mark_live["hist"][i]
            hist.append((float(mcx), float(mcy)))
            if len(hist) > MIC_MARK_SMOOTH_N:
                del hist[0]
            n = len(hist)
            _mic_mark_live["n"][i] = n
            if n < 3:
                any_collecting = True
                continue
            arr = np.array(hist)
            smoothed = arr.mean(axis=0)
            jitter_px = float(np.hypot(*(arr - smoothed).T).std())
            _mic_mark_live["smoothed"][i] = smoothed
            _mic_mark_live["jitter_px"][i] = jitter_px
            locked = (n >= MIC_MARK_SMOOTH_N) and (jitter_px < MIC_MARK_MAX_JITTER_PX)
            _mic_mark_live["locked"][i] = locked
            if not locked:
                any_collecting = True
        parts = []
        for i in (0, 1):
            if _mic_mark_live["locked"][i]:
                parts.append("mark%d LOCKED jitter=%.1fpx" % (i, _mic_mark_live["jitter_px"][i]))
            elif _mic_mark_live["n"][i] > 0:
                parts.append("mark%d %d/%d jitter=%s" % (
                    i, _mic_mark_live["n"][i], MIC_MARK_SMOOTH_N,
                    ("%.1fpx" % _mic_mark_live["jitter_px"][i]) if _mic_mark_live["jitter_px"][i] is not None else "?"))
        _mic_mark_live["status_msg"] = "mic marks: " + (", ".join(parts) if parts else "none seen yet")

    def update_mic_target_live(rec):
        """2026-10-05: EXACT same rolling-average/jitter/sticky-lock/freeze-
        once-locked pattern as update_mic_mark_live() above, but for the
        single system-computed circle-target point (rec["target_cx/cy"])
        instead of 2 independently-tracked ink marks. Only called when
        MIC_RCM_MODE=="circle". See that function's docstring for the full
        rationale (sticky through single-frame occlusion, freeze on lock)."""
        if not rec["ok"] or rec["occluded"] or rec.get("target_cx") is None:
            _mic_target_live["status_msg"] = (
                "circle target: frame occluded/unreliable or no radius available "
                "yet (need >=1 ink mark visible to self-calibrate radius), holding last lock")
            return
        if _mic_target_live["locked"]:
            return  # frozen, same reasoning as the per-mark freeze above
        hist = _mic_target_live["hist"]
        hist.append((float(rec["target_cx"]), float(rec["target_cy"])))
        if len(hist) > MIC_MARK_SMOOTH_N:
            del hist[0]
        n = len(hist)
        _mic_target_live["n"] = n
        if n < 3:
            _mic_target_live["status_msg"] = "circle target: %d/%d samples" % (n, MIC_MARK_SMOOTH_N)
            return
        arr = np.array(hist)
        smoothed = arr.mean(axis=0)
        jitter_px = float(np.hypot(*(arr - smoothed).T).std())
        _mic_target_live["smoothed"] = smoothed
        _mic_target_live["jitter_px"] = jitter_px
        locked = (n >= MIC_MARK_SMOOTH_N) and (jitter_px < MIC_MARK_MAX_JITTER_PX)
        _mic_target_live["locked"] = locked
        _mic_target_live["status_msg"] = "circle target: %s jitter=%.1fpx (r=%.1fpx @ %.1fdeg)" % (
            "LOCKED" if locked else "%d/%d" % (n, MIC_MARK_SMOOTH_N), jitter_px,
            rec.get("target_radius_px") or float("nan"), MIC_RCM_ANGLE_DEG or float("nan"))

    def _render_mic_overlay(img, rec):
        """2026-10-05: factored out of get_mic_preview() so the live-ROS
        path and the disk-polling fallback path share IDENTICAL overlay
        drawing code (plug/ring circles, mark0/1 dots, the circle-mode
        target point, the OCCLUDED banner, and the "RCM LOCKED" PiP
        thumbnail banner) instead of two copies that could silently drift
        apart -- this was previously inline in get_mic_preview() itself,
        before there were two call sites. Pure function of (img, rec) plus
        the already-updated _mic_mark_live/_mic_target_live globals;
        has no side effects of its own (update_mic_mark_live/
        update_mic_target_live must already have been called on `rec`
        by the caller BEFORE this runs)."""
        # Figure-style inset: geometry is drawn on a 2x supersampled copy of
        # the thumbnail with thin solid strokes, then area-downsampled, so
        # lines stay crisp (~1.5px) instead of aliasing into dashes.
        TW = int(MIC_THUMB_W["w"])
        SS = 2
        f_ = TW / 260.0                   # geometry scale
        fl_ = 1.0 + (f_ - 1.0) * 0.55     # stroke / text scale (grows slower)
        s_ = TW * SS / float(img.shape[1])
        big = cv2.resize(img, (TW * SS, int(round(img.shape[0] * s_))), interpolation=cv2.INTER_AREA)

        def _p(x, y):
            return (int(round(x * s_)), int(round(y * s_)))

        def _stroke_ring(c, r_src, col, w=None):
            w = int(round(3 * fl_)) if w is None else w
            r = int(round(r_src * s_))
            shade = np.zeros_like(big)
            cv2.circle(shade, c, r, (255, 255, 255), w + 3, cv2.LINE_AA)
            m = shade[:, :, :1].astype(np.float32) / 255.0 * 0.45
            big[:] = (big.astype(np.float32) * (1.0 - m)).astype(np.uint8)
            cv2.circle(big, c, r, col, w, cv2.LINE_AA)

        def _dot(c, col, r=5):
            r = int(round(r * f_))
            cv2.circle(big, c, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(big, c, r, col, -1, cv2.LINE_AA)
            cv2.circle(big, c, r + 2, (245, 245, 245), 1, cv2.LINE_AA)

        def _target(c, col, locked):
            R_ = int(round(15 * f_))
            cv2.circle(big, c, R_, (0, 0, 0), int(round(4 * fl_)), cv2.LINE_AA)
            cv2.circle(big, c, R_, col, int(round(2 * fl_)), cv2.LINE_AA)
            cv2.circle(big, c, int(round(4 * fl_)), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(big, c, int(round(3 * fl_)), col, -1, cv2.LINE_AA)

        if MIC_RCM_MODE == "circle":
            _cur_locked = _mic_target_live["locked"]
            _cur_smoothed = _mic_target_live["smoothed"]
        else:
            _cur_locked = _mic_mark_live["locked"].get(MIC_TARGET_MARK, False)
            _cur_smoothed = _mic_mark_live["smoothed"].get(MIC_TARGET_MARK)

        target_px = None
        geo = None
        if rec["ok"]:
            marks = []
            for i in range(rec["n_marks"]):
                mcx, mcy = rec.get("mark%d_cx" % i), rec.get("mark%d_cy" % i)
                if mcx is None:
                    continue
                smoothed_i = _mic_mark_live["smoothed"].get(i)
                if smoothed_i is not None:
                    mcx, mcy = float(smoothed_i[0]), float(smoothed_i[1])
                if (MIC_RCM_MODE == "mark") and (i == MIC_TARGET_MARK):
                    target_px = (mcx, mcy)
                    continue
                marks.append((mcx, mcy))
            ring_r = rec["target_radius_px"] if (MIC_RCM_MODE == "circle" and rec.get("target_cx") is not None) \
                else rec["ring_outer_r"]
            geo = {"cx": float(rec["plug_cx"]), "cy": float(rec["plug_cy"]),
                   "plug_r": float(rec["plug_r"]), "ring_r": float(ring_r), "marks": marks}
            if MIC_RCM_MODE == "circle" and rec.get("target_cx") is not None:
                target_px = (rec["target_cx"], rec["target_cy"])
        fz = _mic_geo_freeze
        if _cur_locked:
            if fz["geo"] is None:
                buf = fz["buf"] + ([geo] if geo is not None else [])
                if buf:
                    med = lambda key: float(np.median([g[key] for g in buf]))
                    fz["geo"] = {"cx": med("cx"), "cy": med("cy"), "plug_r": med("plug_r"),
                                 "ring_r": med("ring_r"), "marks": buf[-1]["marks"]}
            if fz["geo"] is not None:
                geo = fz["geo"]
        else:
            if fz["geo"] is not None:
                fz["buf"] = []
            fz["geo"] = None
            if geo is not None:
                fz["buf"].append(geo)
                del fz["buf"][:-12]
        if geo is not None:
            pc = _p(geo["cx"], geo["cy"])
            _stroke_ring(pc, geo["plug_r"], MIC_C_LIMBUS)
            _stroke_ring(pc, geo["ring_r"], MIC_C_MARKRING)
            for (mx_, my_) in geo["marks"]:
                _dot(_p(mx_, my_), (235, 235, 235), 4)
        if _cur_smoothed is not None:
            target_px = (float(_cur_smoothed[0]), float(_cur_smoothed[1]))
        if target_px is not None and (rec["ok"] or _cur_locked):
            _target(_p(*target_px), MIC_C_LOCK if _cur_locked else MIC_C_SEEK, _cur_locked)

        thumb = cv2.resize(big, (TW, big.shape[0] // SS), interpolation=cv2.INTER_AREA)
        k_ = TW / float(img.shape[1])
        avoid = []
        if geo is not None:
            for (mx_, my_) in geo["marks"]:
                avoid.append((int(mx_ * k_ - 12 * f_), int(my_ * k_ - 12 * f_),
                              int(mx_ * k_ + 12 * f_), int(my_ * k_ + 12 * f_)))

        if target_px is not None and (rec["ok"] or _cur_locked):
            tx, ty = target_px[0] * k_, target_px[1] * k_
            col = MIC_C_LOCK if _cur_locked else MIC_C_SEEK
            lw = hud_tw("RCM", 0.36 * fl_, True)
            if geo is not None:
                vx, vy = tx - geo["cx"] * k_, ty - geo["cy"] * k_
            else:
                vx, vy = 1.0, -1.0
            vn = max(1e-6, (vx * vx + vy * vy) ** 0.5)
            vx, vy = vx / vn, vy / vn                      # outward from ring centre
            ex, ey = tx + vx * (8 * f_), ty + vy * (8 * f_)
            kx, ky = tx + vx * (24 * f_), ty + vy * (24 * f_)
            kx = min(max(kx, 10 if vx < 0 else 4), TW - 8 - (0 if vx < 0 else lw))
            ky = min(max(ky, 14 * fl_), thumb.shape[0] - 26 * fl_)
            cv2.line(thumb, (int(ex), int(ey)), (int(kx), int(ky)), (0, 0, 0), int(round(2 * fl_)) + 1, cv2.LINE_AA)
            cv2.line(thumb, (int(ex), int(ey)), (int(kx), int(ky)), col, max(1, int(round(fl_))), cv2.LINE_AA)
            lx_ = int(kx + 3) if vx >= 0 else int(kx - 3 - lw)
            hud_text(thumb, "RCM", (lx_, int(ky + 4 * fl_)), 0.36 * fl_, col, bold=True, shadow=True)
            avoid.append((int(tx - 22 * f_), int(ty - 22 * f_), int(tx + 22 * f_), int(ty + 22 * f_)))
            avoid.append((lx_ - 4, int(ky - 12 * fl_), lx_ + lw + 4, int(ky + 8 * fl_)))

        def _ring_label(cx_t, cy_t, r_t, text, col):
            """Contour label: a small pill sitting ON the ring, at the first
            free position around it."""
            tw_ = hud_tw(text, 0.34 * fl_, True)
            th_ = int(15 * fl_)
            pad = int(5 * fl_)
            for ang in (90, 68, 112, 46, 134, 24, 156, 4, 176, -26, -154, -90):
                a_ = np.radians(ang)
                x_ = cx_t + r_t * np.cos(a_)
                y_ = cy_t - r_t * np.sin(a_)
                rc = (int(x_ - tw_ / 2.0 - pad), int(y_ - th_ / 2.0), int(x_ + tw_ / 2.0 + pad), int(y_ + th_ / 2.0))
                if rc[0] < 3 or rc[2] > thumb.shape[1] - 3 or rc[1] < 3 or rc[3] > thumb.shape[0] - int(22 * fl_):
                    continue
                if any(not (rc[2] < o[0] or rc[0] > o[2] or rc[3] < o[1] or rc[1] > o[3]) for o in avoid):
                    continue
                hud_panel(thumb, rc[0], rc[1], rc[2], rc[3], 0.62)
                cv2.rectangle(thumb, (rc[0], rc[1]), (rc[2], rc[3]), col, 1, cv2.LINE_AA)
                hud_text(thumb, text, (rc[0] + pad, rc[3] - int(4 * fl_)), 0.34 * fl_, col, bold=True)
                avoid.append(rc)
                return

        if geo is not None:
            _ring_label(geo["cx"] * k_, geo["cy"] * k_, geo["plug_r"] * k_, "Limbus", MIC_C_LIMBUS)
            _ring_label(geo["cx"] * k_, geo["cy"] * k_, geo["ring_r"] * k_,
                        "Marker ring" if MIC_RCM_MODE == "circle" else "Outer ring", MIC_C_MARKRING)
        if _cur_locked and _cur_smoothed is not None and os.path.isfile(MIC_EXTRINSIC_CALIB_NPZ):
            try:
                _calib_disp = np.load(MIC_EXTRINSIC_CALIB_NPZ)
                _xy_disp = _calib_disp["A"].dot(np.array([_cur_smoothed[0], _cur_smoothed[1], 1.0]))
                mm_line = "x %.1f   y %.1f   z %.1f  mm" % (
                    _xy_disp[0], _xy_disp[1], float(_calib_disp["z_plane_mm"]))
                hh_ = int(17 * fl_)
                hud_panel(thumb, 0, thumb.shape[0] - hh_, thumb.shape[1], thumb.shape[0], 0.55)
                hud_text(thumb, mm_line, (int(6 * fl_), thumb.shape[0] - int(5 * fl_)), 0.34 * fl_, HUD_TXT)
            except Exception:
                pass
        return thumb

    def get_mic_preview():
        # 2026-10-05: prefer the direct in-process ROS subscription
        # (MIC_ROS_LIVE, see _MicRosLive) over the disk-polling relay below
        # whenever it has a frame fresher than MIC_ROS_STALE_S -- this is
        # the ACTUAL real-time path (native camera rate, no JPEG round
        # trip); the disk-polling path below is now only the fallback for
        # when MIC_ROS_LIVE=0 or the ROS topic has gone quiet. Handled as
        # its own self-contained branch (not interleaved with the disk
        # path) specifically so a "fresh ROS frame, but still within this
        # tick's reprocess-throttle cooldown" state never falls through to
        # disk polling and risks mixing a newer cached ROS thumb with an
        # older/unrelated disk file -- in that case we just return
        # whatever is already cached, exactly as the disk-only path always
        # did when nothing changed.
        if _mic_ros_live is not None:
            live_frame, live_t, live_n = _mic_ros_live.get_latest()
            if live_frame is not None and (time.time() - live_t) <= MIC_ROS_STALE_S:
                last_ros_t = _mic_prev.get("ros_t")
                if last_ros_t is None or (live_t - last_ros_t) >= MIC_ROS_PROCESS_INTERVAL_S:
                    latest = "ros_live_frame_%d.jpg" % live_n  # synthetic, keeps .jpg-ish downstream display
                    sys.path.insert(0, MIC_AUTONOMY_DIR)
                    import mic_perception as _mp
                    img = live_frame
                    _mic_prev["ros_t"] = live_t
                    rec = _mp.locate(img,
                                      target_angle_deg=(MIC_RCM_ANGLE_DEG if MIC_RCM_MODE == "circle" else None),
                                      target_radius_px=MIC_RCM_RADIUS_PX)
                    update_mic_mark_live(rec)
                    if MIC_RCM_MODE == "circle":
                        update_mic_target_live(rec)
                    thumb = _render_mic_overlay(img, rec)
                    _mic_prev.update(path=latest, mtime=live_t, thumb=thumb, rec=rec)
                return _mic_prev["thumb"], _mic_prev["rec"], _mic_prev["path"], _mic_prev["mtime"]
            # live frame missing/stale -- fall through to disk polling below,
            # same as if MIC_ROS_LIVE were never enabled at all this tick.

        # 2026-10-05: FileNotFoundError race fix -- mic_live_capture_ros.py
        # writes a new still every ~0.5s AND rotates/deletes old ones, so a
        # file that was present during os.listdir() above can vanish before
        # os.path.getmtime() reads it a moment later (confirmed live: this
        # crashed the WHOLE script via an uncaught exception out of
        # log_ring_offset_sample() -> get_mic_preview(), killing an
        # in-progress ring-offset-sample collection run). The race window is
        # a few ms, so one immediate retry on the listdir+stat sequence is
        # enough in practice; only give up (return None, same as "no stills
        # yet") if it fails twice in a row.
        now_scan = time.time()
        if (now_scan - _mic_prev["dir_scan_t"]) < MIC_DIR_SCAN_INTERVAL_S and _mic_prev["path"] is not None:
            latest, mtime = _mic_prev["path"], _mic_prev["mtime"]
        else:
            latest = None
            mtime = None
            for _attempt in range(2):
                try:
                    imgs = [os.path.join(MIC_LIVE_STILLS_DIR, f) for f in os.listdir(MIC_LIVE_STILLS_DIR)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if not imgs:
                        return None
                    latest = max(imgs, key=os.path.getmtime)
                    mtime = os.path.getmtime(latest)
                    _mic_prev["dir_scan_t"] = now_scan
                    break
                except OSError:
                    latest = None
                    continue
            if latest is None:
                return None
        # 2026-10-06: found via `sudo py-spy dump` alongside the ring-detect
        # throttle above -- mic_perception.locate() (HoughCircles +
        # detect_entry_marks' radial profile scan) was ALSO repeatedly
        # caught live on the main thread. It was already gated on "did the
        # file actually change" (good), but mic_live_capture_ros.py writes
        # a new still every ~0.5s unconditionally, so on this contended
        # machine that was still enough to stall grab_show() roughly twice
        # a second. Add an explicit wall-clock floor on top of the
        # file-change check -- same cached-result-until-due pattern as
        # RING_DETECT_MIN_INTERVAL_S -- so a burst of new stills (or a
        # future faster capture interval) can't reprocess faster than this
        # even if every single one is "new".
        # 2026-10-06 (pt.2): non-blocking pickup of whatever the background
        # worker finished since the last grab_show() tick -- this is the
        # ONLY place that touches _mic_future/_mic_prev from the main
        # thread for the disk path now, and it never waits.
        if _mic_future[0] is not None and _mic_future[0].done():
            fut, _mic_future[0] = _mic_future[0], None
            try:
                img, rec = fut.result()
            except Exception as e:
                print("  !! mic_perception.locate() failed in background (non-fatal): %s" % e)
            else:
                update_mic_mark_live(rec)
                if MIC_RCM_MODE == "circle":
                    update_mic_target_live(rec)
                thumb = _render_mic_overlay(img, rec)
                _mic_prev.update(path=_mic_prev.get("_pending_path"), mtime=_mic_prev.get("_pending_mtime"),
                                  thumb=thumb, rec=rec, locate_t=time.time())
        _now_mp = time.time()
        _mic_due = (_now_mp - _mic_prev.get("locate_t", 0.0)) >= MIC_PROCESS_MIN_INTERVAL_S
        if (_mic_prev["path"] != latest or _mic_prev["mtime"] != mtime) and _mic_due and _mic_future[0] is None:
            _mic_prev["_pending_path"] = latest
            _mic_prev["_pending_mtime"] = mtime
            _mic_future[0] = _mic_executor.submit(
                _mic_bg_load_and_locate, latest,
                (MIC_RCM_ANGLE_DEG if MIC_RCM_MODE == "circle" else None), MIC_RCM_RADIUS_PX)
        # 2026-10-06 (pt.2): the background job above has not necessarily
        # completed yet (e.g. the very first still this session) -- callers
        # (draw()) expect either a fully-populated 4-tuple or exactly None,
        # same contract the old fully-synchronous code always satisfied.
        if _mic_prev["thumb"] is None:
            return None
        return _mic_prev["thumb"], _mic_prev["rec"], _mic_prev["path"], _mic_prev["mtime"]

    def try_lock_microscope_rcm_live():
        """'y': the SAME correction as 'e', collapsed into ONE key press for
        use right before the final insertion move (not just once at session
        start). Rationale (2026-09-21): the D405 depth-camera pipeline's
        3D estimate of the entry point is the main source of tip-arrival
        error -- NOT the robot's own kinematics/encoders, which are already
        trusted (that is what 'g'/'t' rely on to reach whatever target they
        are given, and cur_tip() is built on that same trusted FK). So this
        does NOT try to visually track the needle tip in the microscope
        image (unreliable -- see mic_perception.py's docstring: the
        instrument shows up as a static, out-of-focus blob in every dataset
        collected so far). Instead it re-registers WHERE THE TARGET ACTUALLY
        IS using the microscope (which resolves the entry marks far more
        precisely than the D405), then lets the robot's own trusted
        kinematics drive the tip there -- exactly like 'z'/'e' already do,
        just triggered from a fresh photo in one step instead of a separate
        terminal command + 'e'.

        Requires, both one-time (from the lab calibration protocol in
        microscope_autonomy/README.md):
          - MIC_EXTRINSIC_CALIB_NPZ already fit (mic_extrinsic_calib.py --fit)
        and every time this key is pressed:
          - a FRESH, clean (unoccluded, in-focus) microscope still of the
            entry point. ***CORRECTED 2026-10-05*** -- the paragraph below
            describing a manual single-still transfer workflow was true as
            of 2026-09-21 but is now STALE: there IS a genuine real-time
            video feed (DeckLink SDI capture card -> gscam_decklink.launch
            -> /decklink/camera/image_raw @ ~60Hz). Two ways this function
            gets a fresh frame now:
              (a) MIC_ROS_LIVE=1 (new, preferred): this script subscribes to
                  that ROS topic DIRECTLY, in-process, no disk I/O at all --
                  see _MicRosLive / get_mic_preview()'s top branch. "fresh"
                  just means "a frame arrived in the last MIC_ROS_STALE_S".
              (b) MIC_ROS_LIVE=0 (fallback, original path, kept for when the
                  topic isn't up): mic_live_capture_ros.py runs as a
                  SEPARATE process, subscribes to the same topic, and writes
                  a throttled (default 0.5s/2Hz) JPEG relay into
                  MIC_LIVE_STILLS_DIR -- which is what the rest of this
                  paragraph originally described. You do NOT manually
                  transfer anything in EITHER case anymore; that was only
                  ever true before the DeckLink card was wired up.
            Either way, it warns loudly if the frame is older than
            MIC_LIVE_MAX_AGE_S seconds (staleness gate disabled by default
            as of 2026-09-21, see that constant's definition -- re-enable
            with a real seconds value before lab use) so a stuck/dead
            pipeline can't silently get reused.
        """
        if not os.path.isfile(MIC_EXTRINSIC_CALIB_NPZ):
            S.msg = "no %s -- run mic_extrinsic_calib.py --fit first (one-time lab calibration)" % MIC_EXTRINSIC_CALIB_NPZ
            return False
        preview = get_mic_preview()
        if preview is None:
            S.msg = "no stills in %s -- capture+transfer one first" % MIC_LIVE_STILLS_DIR
            return False
        _thumb, rec, latest, mtime = preview
        age_s = time.time() - mtime
        if age_s > MIC_LIVE_MAX_AGE_S:
            S.msg = ("newest still in %s is %.0fs old (>%.0fs limit) -- %s -- capture a FRESH "
                      "one, don't reuse an old photo" % (MIC_LIVE_STILLS_DIR, age_s,
                      MIC_LIVE_MAX_AGE_S, os.path.basename(latest)))
            print("  " + S.msg)
            return False
        if not rec["ok"]:
            S.msg = "mic_perception: plug/limbus not found in %s -- retake" % os.path.basename(latest)
            return False
        # 2026-10-05: branch on MIC_RCM_MODE -- "mark" is the EXACT
        # pre-2026-10-05 behavior (literal ink-dot MIC_TARGET_MARK), unchanged
        # below; "circle" instead targets rec["target_cx/cy"] (the system-
        # computed point at MIC_RCM_ANGLE_DEG around the limbus, radius
        # self-calibrated from whichever marks are visible this frame -- see
        # mic_perception.point_on_plug_circle). Everything downstream (the
        # lock-gate, the affine transform to mm, the jitter accounting, the
        # final _vision_rcm write) is IDENTICAL in structure either way; only
        # which px-space point/lock-state feeds it differs.
        if MIC_RCM_MODE == "circle":
            target_label = "circle@%.0fdeg" % MIC_RCM_ANGLE_DEG
            cx, cy = rec.get("target_cx"), rec.get("target_cy")
            if cx is None:
                S.msg = ("mic_perception: no limbus radius available in %s (need >=1 ink mark "
                          "visible to self-calibrate radius, or set MIC_RCM_RADIUS_PX) -- retake"
                          % os.path.basename(latest))
                return False
            tgt_locked = _mic_target_live["locked"]
            n_i = _mic_target_live["n"]
            jit_i = _mic_target_live["jitter_px"]
            if not tgt_locked:
                S.msg = ("%s not stable yet: %d/%d samples, jitter=%s (need <%.1fpx) -- "
                          "hold still a moment, don't press 'y' yet" % (
                              target_label, n_i, MIC_MARK_SMOOTH_N,
                              ("%.1fpx" % jit_i) if jit_i is not None else "?", MIC_MARK_MAX_JITTER_PX))
                print("  " + S.msg)
                return False
            smoothed_px = _mic_target_live["smoothed"]
            hist_px = np.array(_mic_target_live["hist"])
            m = MIC_TARGET_EDGE_MARGIN_PX
            if not (m <= smoothed_px[0] <= MIC_FRAME_W - m and m <= smoothed_px[1] <= MIC_FRAME_H - m):
                S.msg = ("%s target (%.0f, %.0f)px is outside the microscope view -- the hole could not "
                          "be measured afterwards; reposition the phantom (mark0 on the left) and retry"
                          % (target_label, smoothed_px[0], smoothed_px[1]))
                print("  " + S.msg)
                return False
            th_m = rec.get("mark%d_theta_deg" % MIC_TARGET_MARK)
            if MIC_ANGLE_REL_MARK and th_m is not None and rec.get("plug_cx") is not None:
                th_t = np.degrees(np.arctan2(smoothed_px[1] - rec["plug_cy"], smoothed_px[0] - rec["plug_cx"]))
                rel = (th_t - th_m + 180.0) % 360.0 - 180.0
                if abs((rel - MIC_RCM_ANGLE_DEG + 180.0) % 360.0 - 180.0) > 5.0:
                    S.msg = ("%s: locked target sits at %+.1f deg from mark%d but this trial needs %+.1f deg "
                              "-- stale lock; press SHIFT+M (or 'h') and retry" % (
                                  target_label, rel, MIC_TARGET_MARK, MIC_RCM_ANGLE_DEG))
                    print("  " + S.msg)
                    return False
        else:
            target_label = "mark%d" % MIC_TARGET_MARK
            cx, cy = rec.get("mark%d_cx" % MIC_TARGET_MARK), rec.get("mark%d_cy" % MIC_TARGET_MARK)
            if cx is None:
                S.msg = ("mic_perception: mark%d not found in %s (n_marks=%d) -- retake, or check "
                          "MIC_TARGET_MARK" % (MIC_TARGET_MARK, os.path.basename(latest), rec["n_marks"]))
                return False

            # 2026-09-28: gate on _mic_mark_live's rolling lock (see its
            # docstring / MIC_MARK_SMOOTH_N) instead of trusting this one raw
            # frame's centroid outright -- measured live jitter on a single
            # frame was ~3.7px RMS / ~9px max, which is exactly the "green dot
            # flies around" symptom. Mirrors 'z'/'x' requiring _ring_live to be
            # locked before they'll return a point.
            mark_locked = _mic_mark_live["locked"].get(MIC_TARGET_MARK, False)
            n_i = _mic_mark_live["n"].get(MIC_TARGET_MARK, 0)
            jit_i = _mic_mark_live["jitter_px"].get(MIC_TARGET_MARK)
            if not mark_locked:
                S.msg = ("mark%d not stable yet: %d/%d samples, jitter=%s (need <%.1fpx) -- "
                          "hold still a moment, don't press 'y' yet" % (
                              MIC_TARGET_MARK, n_i, MIC_MARK_SMOOTH_N,
                              ("%.1fpx" % jit_i) if jit_i is not None else "?", MIC_MARK_MAX_JITTER_PX))
                print("  " + S.msg)
                return False
            smoothed_px = _mic_mark_live["smoothed"][MIC_TARGET_MARK]
            hist_px = np.array(_mic_mark_live["hist"][MIC_TARGET_MARK])

        calib = np.load(MIC_EXTRINSIC_CALIB_NPZ)
        A = calib["A"]
        xy = A.dot(np.array([smoothed_px[0], smoothed_px[1], 1.0]))
        p = np.array([xy[0], xy[1], float(calib["z_plane_mm"])], np.float64)
        calib_rms = float(np.sqrt((calib["resid_mm"] ** 2).mean()))
        # honest combined jitter: transform the SAME px history window
        # through the real affine A (not a guessed px->mm scale) and take
        # the xy spread -- this is the temporal mark-tracking jitter, kept
        # separate from calib_rms (the one-time extrinsic-fit residual) so
        # neither number hides the other, same "don't collapse two
        # different error sources into one misleading value" principle as
        # _ring_live's diam-vs-depth agreement check.
        hist_mm = (A[:, :2].dot(hist_px.T) + A[:, 2:3]).T
        mark_jitter_mm = float(np.linalg.norm(hist_mm - hist_mm.mean(axis=0), axis=1).std())

        _vision_rcm["p"] = p; _vision_rcm["t"] = time.time()
        _vision_rcm["n"] = n_i
        _vision_rcm["jitter"] = max(calib_rms, mark_jitter_mm)
        S.rcm_source = "vision"
        S.msg = ("MICROSCOPE RCM LIVE-locked %s [%.2f %.2f %.2f]mm  "
                  "mark_jitter~%.3fmm(n=%d)  calib_rms~%.3fmm  photo %.0fs old (%s)%s -- "
                  "VERIFY before 'g'/'t'!" % (
                      target_label, p[0], p[1], p[2], mark_jitter_mm, n_i, calib_rms,
                      age_s, os.path.basename(latest),
                      "  [FLAGGED OCCLUDED]" if rec["occluded"] else ""))
        print("  " + S.msg)
        return True

    def d405_plane_z_at(xy, ring_base):
        """Height of the phantom face at base-frame xy: plane through the D405 estimate of
        mark0 (ring + offset_local_mark0 rotated into the ring-plane frame) with the ring-plane
        normal. None if there is no fresh ring-plane normal for THIS phantom pose."""
        if PHANTOM_NORMAL is None:
            return None
        if np.linalg.norm(np.asarray(PHANTOM_NORMAL["point"]) - ring_base) > MIC_Z_NORMAL_MAX_AGE_MM:
            print("  !! phantom normal was estimated at a different phantom pose -- press 'n' "
                  "after 'z'; using the fixed-plane z this time")
            return None
        try:
            off_local = np.array(np.load(RING_TO_ENTRY_OFFSET_NPZ)["offset_local_mark0"], np.float64)
        except (KeyError, OSError):
            return None
        n = PHANTOM_NORMAL["normal"] / np.linalg.norm(PHANTOM_NORMAL["normal"])
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, n)
        s, c = np.linalg.norm(v), float(z.dot(n))
        R = np.eye(3)
        if s > 1e-9:
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + K + K.dot(K) * ((1 - c) / s ** 2)
        p0 = ring_base + R.dot(off_local)
        return float(p0[2] - (n[0] * (xy[0] - p0[0]) + n[1] * (xy[1] - p0[1])) / n[2])

    def try_correspondence_check():
        """'`' (backtick -- deliberately unshifted/far from any jog letter,
        see the 2026-10-01 Shift-key safety incident this project already
        hit twice with new keybindings; EVERY lowercase letter is already
        bound in this script, see the grep that found this one free):

        2026-10-05: the ONE combined action for the pragmatic-ISMR ask
        ("make the microscope RCM point and the depth-camera RCM point be
        verifiably the SAME point, across >=10 closed-loop trials, with
        minimal operator steps"). Previously this needed 2 separate key
        presses ('0' to log a paired sample, 'y' to actually lock+drive the
        trial from the microscope) plus a mental/offline comparison -- this
        key does ALL of it in one press:
          1. computes the D405 ring-based estimate (raw AND, if available,
             corrected via whichever offset model MIC_RCM_MODE selects --
             mirrors 'x' exactly, see try_lock_vision_rcm_with_offset(),
             but WITHOUT mutating _vision_rcm -- the microscope stays the
             one true source that actually drives 'g'/'t'/'i'/'p'/'b',
             exactly matching this project's established "D405 for rough
             guidance, microscope for the real number" principle)
          2. calls try_lock_microscope_rcm_live() (same as 'y' -- gates on
             staleness/occlusion/mark-or-circle-lock exactly as before, and
             on success DOES set _vision_rcm to the microscope point, so
             this key can fully replace 'y' for driving the upcoming trial)
          3. computes discrepancy = |mic_point - d405_point|, prints it
             PROMINENTLY (this is the number the whole campaign exists to
             report), and appends ONE row to BOTH correspondence_log.csv
             (clean, fixed-schema, trivial to describe() for n>=10 mean/std)
             AND ring_offset_samples.jsonl (so --fit/--fit-circle/--compare
             keep working off the exact same button press, no separate '0').

        Does NOT require an offset to already exist -- works from trial #1
        (logs discrepancy_raw only, offset_source="UNCORRECTED") so this is
        also how you'd bootstrap the very first ring_offset_samples.jsonl
        entries, exactly like '0' already did.
        """
        stale = [p for p in calib_freshness_problems()
                 if not p.startswith("ring_to_entry_offset_circle MISSING")]
        if stale:
            S.msg = "`: REFUSED (stale calib): " + "; ".join(stale)
            print("  " + S.msg)
            return False
        if not _ring_live["locked"]:
            S.msg = _ring_live["status_msg"] or "ring not locked yet"
            print("  `: not locked yet -- %s" % S.msg)
            return False
        d405_raw = _ring_live["smoothed"].copy()

        d405_corrected = None
        offset_source = "UNCORRECTED (no offset file yet)"
        if MIC_RCM_MODE == "circle" and MIC_CIRCLE_D405_MODEL == "fixed":
            if os.path.isfile(RING_TO_ENTRY_OFFSET_NPZ) and os.path.isfile(MIC_EXTRINSIC_CALIB_NPZ):
                try:
                    off0 = np.array(np.load(RING_TO_ENTRY_OFFSET_NPZ)["offset_mark%d" % MIC_TARGET_MARK], np.float64)
                    A_lin = np.load(MIC_EXTRINSIC_CALIB_NPZ)["A"][:2, :2]
                    th0 = np.radians(MIC_NOMINAL_MARK_THETA_DEG)
                    tht = th0 + np.radians(MIC_RCM_ANGLE_DEG) if MIC_ANGLE_REL_MARK else np.radians(MIC_RCM_ANGLE_DEG)
                    dpx = MIC_NOMINAL_MARK_R_PX * np.array([np.cos(tht) - np.cos(th0), np.sin(tht) - np.sin(th0)])
                    step = np.zeros(3)
                    step[:2] = A_lin.dot(dpx)
                    d405_corrected = d405_raw + off0 + step
                    offset_source = "mark%d_fixed_vector+circle_step(%.0fdeg,[%.2f %.2f]mm)" % (
                        MIC_TARGET_MARK, MIC_RCM_ANGLE_DEG, step[0], step[1])
                except KeyError:
                    pass
        elif MIC_RCM_MODE == "circle":
            # 2026-10-05: MIC_RCM_RADIUS_LOCAL_MM, if set, overrides the
            # npz-fit radius here too -- same rationale as the 'x' key above
            # (try_lock_vision_rcm_with_offset): mark0 vs mark1's independent
            # fits disagreed by ~5mm today, so neither is a validated shared
            # geometric constant; a real clinical radius is preferable when
            # available. This path has NO npz-existence requirement in that
            # case (unlike the npz branch below, which still needs it).
            if PHANTOM_NORMAL is not None and (
                    MIC_RCM_RADIUS_LOCAL_MM is not None
                    or os.path.isfile(RING_TO_ENTRY_OFFSET_CIRCLE_NPZ)):
                try:
                    if MIC_RCM_RADIUS_LOCAL_MM is not None:
                        radius_local = MIC_RCM_RADIUS_LOCAL_MM
                        radius_src_label = "CLINICAL-FIXED"
                    else:
                        off_d = np.load(RING_TO_ENTRY_OFFSET_CIRCLE_NPZ)
                        radius_local = float(off_d["radius_local_mm_mark%d" % MIC_TARGET_MARK])
                        radius_src_label = "npz-fit-mark%d" % MIC_TARGET_MARK
                    sys.path.insert(0, MIC_AUTONOMY_DIR)
                    import mic_extrinsic_calib as _mec
                    normal = PHANTOM_NORMAL["normal"] / np.linalg.norm(PHANTOM_NORMAL["normal"])
                    u, v = _mec.tangent_frame_from_normal(normal)
                    d405_corrected = (d405_raw
                                       + radius_local * np.cos(np.radians(MIC_RCM_ANGLE_DEG)) * u
                                       + radius_local * np.sin(np.radians(MIC_RCM_ANGLE_DEG)) * v)
                    offset_source = "circle(radius_local=%.2fmm[%s],angle=%.0fdeg)" % (
                        radius_local, radius_src_label, MIC_RCM_ANGLE_DEG)
                except KeyError:
                    pass
        else:
            if os.path.isfile(RING_TO_ENTRY_OFFSET_NPZ):
                try:
                    off_d = np.load(RING_TO_ENTRY_OFFSET_NPZ)
                    offset = np.array(off_d["offset_mark%d" % MIC_TARGET_MARK], np.float64)
                    d405_corrected = d405_raw + offset
                    offset_source = "mark%d_fixed_vector" % MIC_TARGET_MARK
                except KeyError:
                    pass

        if not try_lock_microscope_rcm_live():
            S.msg = "`: D405 side computed OK, but microscope side failed -- %s" % S.msg
            print("  " + S.msg)
            return False
        z_src = "mic fixed plane"
        if MIC_Z_SOURCE == "d405plane":
            z_plane = d405_plane_z_at(_vision_rcm["p"][:2], d405_raw)
            if z_plane is not None:
                _vision_rcm["p"] = np.array([_vision_rcm["p"][0], _vision_rcm["p"][1], z_plane])
                z_src = "D405 ring plane"
        mic_p = _vision_rcm["p"].copy()
        print("  RCM z = %.2fmm (%s)" % (mic_p[2], z_src))
        mic_jitter_mm = _vision_rcm["jitter"]
        mic_n = _vision_rcm["n"]
        prev_rec = _mic_prev.get("rec") or {}
        calib_rms_mm = float("nan")
        if os.path.isfile(MIC_EXTRINSIC_CALIB_NPZ):
            try:
                calib_rms_mm = float(np.sqrt((np.load(MIC_EXTRINSIC_CALIB_NPZ)["resid_mm"] ** 2).mean()))
            except Exception:
                pass

        discrepancy_raw_mm = float(np.linalg.norm(mic_p - d405_raw))
        discrepancy_corrected_mm = (float(np.linalg.norm(mic_p - d405_corrected))
                                     if d405_corrected is not None else None)

        _last_correspondence.update({
            "t": time.time(), "d405_raw": d405_raw, "d405_corrected": d405_corrected,
            "mic": mic_p, "disc_raw_mm": discrepancy_raw_mm, "disc_corr_mm": discrepancy_corrected_mm,
            "offset_source": offset_source,
        })

        log_correspondence(
            mic_rcm_mode=MIC_RCM_MODE, mic_target_mark=MIC_TARGET_MARK,
            d405_raw_x_mm=d405_raw[0], d405_raw_y_mm=d405_raw[1], d405_raw_z_mm=d405_raw[2],
            d405_corrected_x_mm=(d405_corrected[0] if d405_corrected is not None else ""),
            d405_corrected_y_mm=(d405_corrected[1] if d405_corrected is not None else ""),
            d405_corrected_z_mm=(d405_corrected[2] if d405_corrected is not None else ""),
            mic_x_mm=mic_p[0], mic_y_mm=mic_p[1], mic_z_mm=mic_p[2],
            discrepancy_raw_mm=round(discrepancy_raw_mm, 4),
            discrepancy_corrected_mm=(round(discrepancy_corrected_mm, 4)
                                       if discrepancy_corrected_mm is not None else ""),
            offset_source=offset_source, ring_jitter_mm=_ring_live["jitter"], ring_n=_ring_live["n"],
            mic_jitter_mm=mic_jitter_mm, mic_calib_rms_mm=calib_rms_mm, mic_n=mic_n,
            occluded=bool(prev_rec.get("occluded")), mic_still_path=_mic_prev.get("path"),
        )
        # ALSO feed the existing ring_offset_samples.jsonl pipeline (same
        # schema log_ring_offset_sample() writes) so this one keypress is a
        # strict superset of pressing '0' separately -- --fit/--fit-circle/
        # --compare on that file keep seeing every sample this key logs too.
        mark_bases = {}
        try:
            sys.path.insert(0, MIC_AUTONOMY_DIR)
            import mic_extrinsic_calib as _mec2
            for i in (0, 1):
                mcx, mcy = prev_rec.get("mark%d_cx" % i), prev_rec.get("mark%d_cy" % i)
                if mcx is not None:
                    mark_bases[i] = list(_mec2.pixel_to_base_xy(mcx, mcy, MIC_EXTRINSIC_CALIB_NPZ))
            if mark_bases:
                out = {
                    "timestamp": time.time(), "timestamp_iso": datetime.datetime.now().isoformat(),
                    "ring_base_mm": d405_raw.tolist(), "ring_jitter_mm": float(_ring_live["jitter"]),
                    "ring_n": int(_ring_live["n"]), "mic_still_path": _mic_prev.get("path"),
                    "occluded": bool(prev_rec.get("occluded")), "calib_source_npz": MIC_EXTRINSIC_CALIB_NPZ,
                    "phantom_normal_base": (PHANTOM_NORMAL["normal"].tolist() if PHANTOM_NORMAL is not None else None),
                    "phantom_normal_point_base": (PHANTOM_NORMAL["point"].tolist() if PHANTOM_NORMAL is not None else None),
                }
                for i in (0, 1):
                    out["mark%d_base_mm" % i] = mark_bases.get(i)
                with open(RING_OFFSET_SAMPLES_LOG, "a") as fh:
                    fh.write(json.dumps(out) + "\n")
        except Exception as e:
            print("  !! %s append failed (non-fatal): %s" % (RING_OFFSET_SAMPLES_LOG, e))

        S.msg = ("CORRESPONDENCE CHECK: mic=[%.2f %.2f %.2f]mm  d405_raw=[%.2f %.2f %.2f]mm  "
                  "discrepancy_raw=%.3fmm%s  (%s)  -> %s + %s" % (
                      mic_p[0], mic_p[1], mic_p[2], d405_raw[0], d405_raw[1], d405_raw[2],
                      discrepancy_raw_mm,
                      ("  discrepancy_CORRECTED=%.3fmm" % discrepancy_corrected_mm
                       if discrepancy_corrected_mm is not None else "  (no correction applied yet)"),
                      offset_source, os.path.basename(CORRESPONDENCE_LOG),
                      os.path.basename(RING_OFFSET_SAMPLES_LOG)))
        print("  " + S.msg)
        _pose = {"tilt_deg": None, "yaw_diff_deg": None, "rcm_z_src": z_src}
        if PHANTOM_NORMAL is not None:
            _nn = PHANTOM_NORMAL["normal"] / np.linalg.norm(PHANTOM_NORMAL["normal"])
            _pose["tilt_deg"] = float(np.degrees(np.arccos(np.clip(_nn[2], -1.0, 1.0))))
        if 0 in mark_bases and 1 in mark_bases:
            _d01 = np.subtract(mark_bases[1], mark_bases[0])
            _yaw = float(np.degrees(np.arctan2(_d01[1], _d01[0])))
            _pose["yaw_diff_deg"] = (_yaw - PHANTOM_TRAIN_YAW_DEG + 180.0) % 360.0 - 180.0
        _last_correspondence.update(_pose)
        print("  phantom pose: tilt %s, yaw vs training %s (microscope XY target is pose-invariant; "
              "D405 cross-check is calibrated near tilt<10deg, |yaw|<5deg)" % (
                  ("%.1fdeg" % _pose["tilt_deg"]) if _pose["tilt_deg"] is not None else "?",
                  ("%+.1fdeg" % _pose["yaw_diff_deg"]) if _pose["yaw_diff_deg"] is not None else "?"))
        return True

    def compute_goals():
        """hover_goal/insert_goal are placed along the FIXED calibrated
        target direction (desired_insertion_dir(): phantom normal +
        INSERT_ANGLE_DEG + azimuth) whenever available, NOT the current
        instantaneous tip->RCM line -- this is what lets 'g'/APPROACH run
        BEFORE 't'/ORIENT (see docstring pt.10, 2026-08-18 reorder): the
        hover point sits on the correct axis regardless of which direction
        the tip happens to be pointing at HOME. Falls back to the old
        instantaneous-direction behavior only if no target direction is
        available yet (no phantom normal / manual RCM), same as the
        original home_to_rcm_insert.py.
        """
        tip = cur_tip()
        if tip is None or not rcm_ok():
            return False
        rcm = cur_rcm()
        tdir = desired_insertion_dir()
        if tdir is not None:
            u = tdir
        else:
            d = rcm - tip; n = np.linalg.norm(d)
            if n < 1e-6:
                return False
            u = d / n
        S.hover_goal = rcm - u * HOVER_MM
        S.insert_goal = rcm + u * INSERT_DEPTH_MM
        return True

    def do_tunnel():
        """TUNNEL: oblique advance along the CURRENT tip->RCM direction,
        recomputed fresh right here -- NOT assumed to equal the pre-ORIENT
        hover line. This is what makes the pt.14 (2026-08-18) "rotate then
        insert directly, displacement accepted" strategy safe: no matter
        where 't' actually left the tip (small drift or not), TUNNEL always
        aims from THAT real position back through the real RCM, so it can
        never be aiming at a stale/imaginary point. Returns True/False.
        """
        if not rcm_ok():
            S.msg = "RCM not latched"; return False
        surface_rcm = cur_rcm().copy()
        tip_now = cur_tip()
        if tip_now is None:
            S.msg = "no FrameEE, abort TUNNEL"; return False
        u = tip_now - surface_rcm  # note: sign flips below to point INTO tissue
        un = np.linalg.norm(u)
        if un < 1e-6:
            S.msg = "cannot compute TUNNEL goal (tip==RCM?)"; return False
        u = -u / un  # tip->rcm reversed = insertion direction (surface_rcm - tip, normalized)
        tunnel_goal = surface_rcm + u * TUNNEL_DEPTH_MM
        ok = auto_move(tunnel_goal, "TUNNEL")
        if ok:
            S.internal_rcm = cur_tip().copy()
            S.tunneled = True
            S.state = "TUNNELED"
            S.msg = "TUNNEL done -- internal RCM=[%.2f %.2f %.2f], press 'p' to PLUNGE" % tuple(S.internal_rcm)
            print("  " + S.msg)
        else:
            S.tunneled = False
        return ok

    def azimuth_frame(n):
        """pt.16 (2026-08-22): the two FIXED axes that, together with the
        phantom normal n, make orientation fully/deterministically defined
        (n is axis 1; e0 is axis 2; e1=cross(n,e0) is derived, not free).
        Factored out of desired_insertion_dir() so it can also be printed
        once at startup as an explicit, checkable answer to "what are your
        two axes" -- see that function's docstring and module docstring
        pt.16 for the full rationale/honesty note (e0 is an arbitrary-but-
        fixed engineering reference, not yet the anatomical limbus-tangent
        direction, since this phantom has no limbus to detect).
        Returns (e0, e1), both unit vectors, both perpendicular to n.

        FIXED (2026-09-28, pt.23, real bug -- reported symptom: "the pink
        target-direction line on screen is sometimes correct, sometimes the
        correct one rotated 90deg"): the ORIGINAL implementation picked
        ref0=[0,0,1] normally but HARD-SWITCHED to ref0=[1,0,0] whenever
        abs(dot(n, [0,0,1])) crossed 0.99 (~8.1deg from the Z axis) -- a
        genuine discontinuity: two normals only slightly apart, one just
        under and one just over that threshold, get reference vectors ~90deg
        apart, which drags e0 (and therefore the ENTIRE insertion direction)
        ~90deg apart too. Confirmed this is not hypothetical: three real
        phantom-normal measurements from THIS session had n_z =
        0.950 / 0.994 / 0.991 -- i.e. actually straddling the 0.99
        threshold run to run (normal measurement noise alone is enough to
        cross it), which is exactly the reported flip.
        FIX: replace the branching construction with the standard branchless/
        continuous orthonormal-basis-from-a-single-vector formula (Duff,
        Burgess, Christensen, Hery, Kensler, Liani, Villemin, "Building an
        Orthonormal Basis, Revisited", JCGT 2017) -- e0 is now a CONTINUOUS
        function of n everywhere (only the antipodal point n=(0,0,-1) has no
        continuous choice at all, which is a mathematical fact for ANY
        such construction, not something this fix -- or any fix -- can
        avoid; that exact point is never hit in practice here since the
        phantom normal always points roughly toward the camera/robot, never
        straight away from it).
        """
        s = 1.0 if n[2] >= 0.0 else -1.0
        a = -1.0 / (s + n[2])
        b = n[0] * n[1] * a
        e0 = np.array([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]])
        e1 = np.array([b, s + n[1] * n[1] * a, -n[1]])
        return e0, e1

    def desired_insertion_dir():
        """Unit vector (BASE frame) the needle shaft should point along.

        Two-axis (Darboux-frame-style) orientation definition, see module
        docstring pt.16 for the full rationale (Prof. Iulian's "a normal
        vector alone only fixes 2 of 3 rotational DOF" feedback + the
        clinical "insert parallel to the limbus" literature this maps onto):
        the tangent-plane reference direction (which picks the AZIMUTH of
        the INSERT_ANGLE_DEG tilt) can come from two sources:
          - INSERT_AZIMUTH_DEG is None (only via INSERT_AZIMUTH_DEG=legacy/
            auto): legacy single-axis behavior, IDENTICAL to
            home_to_rcm_insert.py -- azimuth = current tip->RCM approach
            direction, projected into the tangent plane. NOT reproducible
            run-to-run; kept only for A/B comparison.
          - INSERT_AZIMUTH_DEG is a number (DEFAULT is "0.0" as of pt.16):
            azimuth is measured from a FIXED reference direction e0_hat
            (global BASE +Z projected into the tangent plane, or +X if the
            normal is too close to +Z) rotated by INSERT_AZIMUTH_DEG around
            the normal (right-hand rule about +normal). (n_hat, e0_hat,
            n_hat x e0_hat) is a complete, deterministic 2-axis-defined
            orthonormal frame -- this makes the target direction depend
            ONLY on the phantom normal and a number you chose -- fully
            reproducible regardless of which direction the tool approached
            from, necessary for claiming a specific, repeatable "30deg at
            Xdeg azimuth" insertion rather than an incidental one. e0_hat
            is a FIXED ENGINEERING reference, not (yet) the anatomical
            "parallel to limbus" direction -- see pt.16's honesty note.

        Still cached per vision-RCM-lock (see baseline docstring for why:
        NEEDLE_AXIS_LOCAL / tip position moves during ORIENT, so recomputing
        tang_ref every call in the legacy auto-azimuth mode would chase a
        moving target -- this is irrelevant in fixed-azimuth mode but kept
        for symmetry/simplicity).
        """
        if PHANTOM_NORMAL is None:
            return None
        cache_key = (_vision_rcm["t"], INSERT_AZIMUTH_DEG)
        if (_target_dir_cache["dir"] is not None and
                (_target_dir_cache["lock_t"], _target_dir_cache["azimuth_deg"]) == cache_key):
            return _target_dir_cache["dir"]
        n = PHANTOM_NORMAL["normal"]
        n = n / np.linalg.norm(n)

        if INSERT_AZIMUTH_DEG is not None:
            e0, e1 = azimuth_frame(n)
            az = np.radians(INSERT_AZIMUTH_DEG)
            tang_ref = np.cos(az) * e0 + np.sin(az) * e1
            # Printed once per fresh RCM lock (this branch only runs on a
            # cache miss, see the early-return above) -- an explicit,
            # checkable record of the pt.16 two-axis frame for whoever asks
            # "how is orientation fully defined" (this is literally the
            # answer: n, e0, and az between them).
            print("  [azimuth frame, pt.16] n_hat=[%.3f %.3f %.3f]  e0_hat=[%.3f %.3f %.3f]  "
                  "az=%.1fdeg -> tang_ref=[%.3f %.3f %.3f]" %
                  (n[0], n[1], n[2], e0[0], e0[1], e0[2], INSERT_AZIMUTH_DEG,
                   tang_ref[0], tang_ref[1], tang_ref[2]))
        else:
            tang_ref = None
            tip = cur_tip()
            if tip is not None and rcm_ok():
                v = cur_rcm() - tip
                nv = np.linalg.norm(v)
                if nv > 1e-6:
                    cand = v / nv
                    if abs(np.dot(cand, n)) < 0.999:
                        tang_ref = cand
            if tang_ref is None:
                ref = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(n, ref)) > 0.99:
                    ref = np.array([1.0, 0.0, 0.0])
                tang_ref = ref

        t_dir = tang_ref - np.dot(tang_ref, n) * n
        tn = np.linalg.norm(t_dir)
        if tn < 1e-9:
            result = -n
        else:
            t_dir = t_dir / tn
            th = np.radians(INSERT_ANGLE_DEG)
            d = -np.cos(th) * n + np.sin(th) * t_dir
            result = d / np.linalg.norm(d)
        _target_dir_cache["dir"] = result
        _target_dir_cache["lock_t"] = _vision_rcm["t"]
        _target_dir_cache["azimuth_deg"] = INSERT_AZIMUTH_DEG
        return result

    def vision_angle_sanity_check_3d(color_bgr, depth_frame, target_dir, rcm_pt):
        """ADVISORY ONLY -- never blocks or alters control. Thin wrapper
        around the top-level measure_shaft_direction_3d(): projects
        target_dir to get a 2D detection prior + a ROI center, converts the
        aligned depth frame to a metres->mm numpy array, calls the
        independent 3D measurement, and turns the result into (msg, dict)
        for printing/overlay/CSV logging. See module docstring pt.4 and
        measure_shaft_direction_3d()'s own docstring for the full method.

        Returns (msg:str, log_fields:dict) or (None, {}) if a fresh vision
        RCM pixel projection isn't available at all (nothing to anchor the
        ROI on) -- a failed/low-confidence *measurement* still returns a
        message + log_fields (with vis3d_ok=False), it just won't have a
        direction; only a total inability to even attempt it returns None.
        """
        if not handeye_ok:
            return None, {}
        rcm_px = base_to_pixel(rcm_pt, R_he, t_he, intr)
        if rcm_px is None:
            return None, {}

        p_far = rcm_pt - target_dir * 15.0
        p_near = rcm_pt + target_dir * 6.0
        px_far = base_to_pixel(p_far, R_he, t_he, intr)
        px_near = base_to_pixel(p_near, R_he, t_he, intr)
        expected_ang_img_deg = None
        if px_far is not None and px_near is not None:
            expected_ang_img_deg = np.degrees(np.arctan2(
                px_near[1] - px_far[1], px_near[0] - px_far[0])) % 180.0

        depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.float64) * depth_scale * 1000.0

        res = measure_shaft_direction_3d(color_bgr, depth_mm, R_he, t_he, intr,
                                          roi_center_px=rcm_px,
                                          expected_ang_img_deg=expected_ang_img_deg)
        log_fields = {
            "vis3d_ok": bool(res.get("ok")),
            "vis3d_n_points": res.get("n_points", ""),
            "vis3d_n_inliers": res.get("n_inliers", ""),
            "vis3d_inlier_ratio": round(res["inlier_ratio"], 3) if "inlier_ratio" in res else "",
            "vis3d_extent_mm": round(res["extent_mm"], 2) if "extent_mm" in res else "",
            "vis3d_rms_mm": round(res["fit_rms_mm"], 3) if "fit_rms_mm" in res else "",
            "vis3d_stability_deg": round(res["stability_deg"], 2) if "stability_deg" in res else "",
            "vis3d_fail_reason": res.get("fail_reason") or "",
        }
        if not res["ok"]:
            msg = "vision-3D check[depth,independent]: unavailable (%s)" % res["fail_reason"]
            return msg, log_fields

        d = res["direction"]
        err_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(d, target_dir)), 0.0, 1.0))))
        unc_deg = res["angular_uncertainty_deg"]
        log_fields["vis3d_err_deg"] = round(err_deg, 3)
        log_fields["vis3d_unc_deg"] = round(unc_deg, 3)
        # raw direction, BASE frame, sign as returned by measure_shaft_direction_3d
        # (ambiguous up to an overall sign -- resolve when reconstructing the EE-
        # local axis offline by picking whichever sign is closer to target_dir/the
        # NEEDLE_AXIS_LOCAL hypothesis being tested for that pose).
        log_fields["vis3d_dx"] = round(float(d[0]), 5)
        log_fields["vis3d_dy"] = round(float(d[1]), 5)
        log_fields["vis3d_dz"] = round(float(d[2]), 5)
        # A measured discrepancy is only meaningful if it clearly exceeds the
        # measurement's OWN uncertainty -- otherwise "CHECK" would just be
        # reporting measurement noise, not a real disagreement.
        status = "OK" if err_deg < max(2.0 * unc_deg, 3.0) else "CHECK"
        # NOTE: we report diff-from-commanded, not a standalone "measured
        # absolute angle" -- a fitted 3D line has a sign ambiguity (d vs -d
        # fit the same points equally well), so there is no unambiguous
        # single "which way is it pointing" without extra assumptions. The
        # diff-from-commanded number (via |dot(d, target_dir)|) is exactly
        # the one quantity that does NOT depend on resolving that ambiguity.
        msg = ("vision-3D check[depth,independent]: diff-from-commanded=%.1fdeg "
               "(meas.unc~%.1fdeg, inliers=%d/%d [%.0f%%], span=%.1fmm, fit-rms=%.2fmm, "
               "stability=%.1fdeg) [%s]" % (
                   err_deg, unc_deg, res["n_inliers"], res["n_points"], 100 * res["inlier_ratio"],
                   res["extent_mm"], res["fit_rms_mm"], res.get("stability_deg", float("nan")),
                   status))
        return msg, log_fields

    def estimate_phantom_normal_from_frame(color_bgr, depth_frame, rcm_base):
        """Local-plane phantom-surface-normal fit -- ported from
        phantom_normal_estimate.py's main(), but as a dict-returning
        function (never raises/exits) that runs on ONE frame from THIS
        script's own already-open pipe instead of a separate process/
        terminal opening the camera again -- see module docstring pt.6 and
        that script's own METHOD HISTORY comment for why this specific
        bowl-only, red-rim-excluded local plane fit (not a global sphere
        fit, not a bigger patch) is the right method here.

        Returns a dict: ok(bool), normal_base, point_base, n_points,
        plane_rms_mm, centroid_disagree_mm, n_red_dropped, fail_reason.
        """
        if not handeye_ok:
            return {"ok": False, "fail_reason": "hand-eye missing"}
        rcm_cam = R_he.T.dot(rcm_base - t_he)
        if rcm_cam[2] <= 1e-3:
            return {"ok": False, "fail_reason": "RCM projects behind the camera"}
        u0 = int(round(intr.fx * rcm_cam[0] / rcm_cam[2] + intr.ppx))
        v0 = int(round(intr.fy * rcm_cam[1] / rcm_cam[2] + intr.ppy))

        lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
        a_star = lab[:, :, 1]
        h, w = depth_frame.get_height(), depth_frame.get_width()

        pts_cam = []
        n_red_dropped = 0
        for dv in range(-NORMAL_PATCH_RADIUS_PX, NORMAL_PATCH_RADIUS_PX + 1):
            for du in range(-NORMAL_PATCH_RADIUS_PX, NORMAL_PATCH_RADIUS_PX + 1):
                if du * du + dv * dv > NORMAL_PATCH_RADIUS_PX * NORMAL_PATCH_RADIUS_PX:
                    continue
                uu, vv = u0 + du, v0 + dv
                if uu < 0 or vv < 0 or uu >= w or vv >= h:
                    continue
                if a_star[vv, uu] > NORMAL_A_STAR_THRESH:
                    n_red_dropped += 1
                    continue
                z = depth_frame.get_distance(uu, vv) * 1000.0
                if z <= 0:
                    continue
                x = (uu - intr.ppx) / intr.fx * z
                y = (vv - intr.ppy) / intr.fy * z
                pts_cam.append((x, y, z))

        pts_cam = np.array(pts_cam, float)
        if len(pts_cam) < NORMAL_MIN_POINTS:
            return {"ok": False, "n_points": len(pts_cam), "n_red_dropped": n_red_dropped,
                    "fail_reason": "only %d non-red depth points (<%d needed, radius %dpx) -- "
                                    "widen NORMAL_PATCH_RADIUS_PX or check A_STAR_THRESH isn't "
                                    "excluding the whole bowl" %
                                    (len(pts_cam), NORMAL_MIN_POINTS, NORMAL_PATCH_RADIUS_PX)}

        centroid = pts_cam.mean(axis=0)
        X = pts_cam - centroid
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        normal_cam = Vt[-1]  # smallest singular value -- plane normal (dual of measure_shaft_direction_3d)
        normal_cam /= np.linalg.norm(normal_cam)
        resid = X.dot(normal_cam)
        plane_rms_mm = float(np.sqrt(np.mean(resid ** 2)))

        view_dir = centroid / np.linalg.norm(centroid)
        if np.dot(normal_cam, view_dir) > 0:
            normal_cam = -normal_cam

        centroid_base = R_he.dot(centroid) + t_he
        centroid_disagree_mm = float(np.linalg.norm(centroid_base - rcm_base))
        # 2026-10-11: split the disagreement in the CAMERA frame. Lateral =
        # is the sampled patch actually centred on the RCM (what this gate
        # exists to catch). Along-ray = ring-centre depth vs bowl-surface
        # depth, which does not change the fitted plane normal at all and
        # was pushing the old single 3D-norm gate over 5mm intermittently.
        d_cam = centroid - rcm_cam
        ray = rcm_cam / np.linalg.norm(rcm_cam)
        depth_off_mm = float(d_cam.dot(ray))
        lateral_mm = float(np.linalg.norm(d_cam - depth_off_mm * ray))
        bad_lat = lateral_mm > NORMAL_MAX_CENTROID_DISAGREE_MM
        bad_dep = abs(depth_off_mm) > NORMAL_MAX_CENTROID_DEPTH_MM
        if bad_lat or bad_dep:
            return {"ok": False, "n_points": len(pts_cam), "n_red_dropped": n_red_dropped,
                    "plane_rms_mm": plane_rms_mm, "centroid_disagree_mm": centroid_disagree_mm,
                    "centroid_lateral_mm": lateral_mm, "centroid_depth_mm": depth_off_mm,
                    "fail_reason": "bowl centroid vs RCM: lateral=%.2fmm (limit %.1f) "
                                    "depth=%+.2fmm (limit %.1f) total=%.2fmm -- %s" % (
                                        lateral_mm, NORMAL_MAX_CENTROID_DISAGREE_MM,
                                        depth_off_mm, NORMAL_MAX_CENTROID_DEPTH_MM,
                                        centroid_disagree_mm,
                                        "patch not centred on RCM (press 'z' again)" if bad_lat
                                        else "RCM depth far from bowl surface")}

        normal_base = R_he.dot(normal_cam)
        normal_base /= np.linalg.norm(normal_base)
        fail_reason = None
        if plane_rms_mm > NORMAL_MAX_PLANE_RMS_MM:
            fail_reason = ("advisory: plane fit RMS %.2fmm > %.1fmm -- patch may straddle a "
                            "curved/edge region, normal may be less accurate than usual" %
                            (plane_rms_mm, NORMAL_MAX_PLANE_RMS_MM))
        return {"ok": True, "normal_base": normal_base, "normal_cam": normal_cam,
                "point_base": rcm_base.copy(), "n_points": len(pts_cam),
                "n_red_dropped": n_red_dropped, "plane_rms_mm": plane_rms_mm,
                "centroid_disagree_mm": centroid_disagree_mm,
                "centroid_lateral_mm": lateral_mm, "centroid_depth_mm": depth_off_mm,
                "fail_reason": fail_reason}

    _trial_capture = {"base": None}

    def _grab_fresh_mic_stills(prefix, t_after):
        """Copy the next TRIAL_STILLS_N microscope stills written after t_after to
        prefix1.jpg, prefix2.jpg, ... Keeps the UI alive while waiting."""
        saved, seen = [], set()
        t_end = time.time() + TRIAL_STILLS_TIMEOUT_S
        while len(saved) < TRIAL_STILLS_N and time.time() < t_end:
            grab_show()
            try:
                cands = sorted(f for f in os.listdir(MIC_LIVE_STILLS_DIR)
                               if f.endswith(".jpg") and f not in seen)
            except OSError:
                cands = []
            for f in cands:
                src = os.path.join(MIC_LIVE_STILLS_DIR, f)
                try:
                    if os.path.getmtime(src) <= t_after:
                        seen.add(f)
                        continue
                    dst = "%s%d.jpg" % (prefix, len(saved) + 1)
                    shutil.copy2(src, dst)
                except OSError:
                    continue
                seen.add(f)
                saved.append({"src": src, "dst": dst, "mtime": os.path.getmtime(dst)})
                if len(saved) >= TRIAL_STILLS_N:
                    break
        return saved

    def capture_trial_post_retract_stills():
        """After RETRACT reaches HOME the tool is out of the microscope field and the
        puncture hole marks where the trocar actually entered the material."""
        base = _trial_capture["base"]
        if base is None or not os.path.isfile(base + ".json"):
            return
        t_done = time.time()
        S.msg = "saving post-RETRACT microscope stills (puncture hole) -- hold still..."
        print("  " + S.msg)
        saved = _grab_fresh_mic_stills(base + "_post", t_done)
        with open(base + ".json") as fh:
            meta = json.load(fh)
        meta["post_retract_stills"] = saved
        meta["post_retract_timestamp"] = t_done
        with open(base + ".json", "w") as fh:
            json.dump(_jsonable(meta), fh, indent=1)
        _trial_capture["base"] = None
        S.msg = ("trial %s: saved %d/%d post-RETRACT microscope stills -- trial complete" %
                 (meta.get("trial"), len(saved), TRIAL_STILLS_N))
        print("  " + S.msg)

    def capture_trial_plunge_stills(plunge_goal):
        """Copy the next TRIAL_STILLS_N microscope stills written AFTER the PLUNGE
        finished (needle still embedded) into TRIAL_STILLS_DIR, plus a JSON sidecar
        with the robot/vision state needed to score the entry point offline."""
        t_done = time.time()
        trial = completed_trial_count() + 1
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(TRIAL_STILLS_DIR, "trial%02d_%s" % (trial, stamp))
        os.makedirs(TRIAL_STILLS_DIR, exist_ok=True)
        S.msg = "trial %d: saving post-PLUNGE microscope stills -- hold still..." % trial
        print("  " + S.msg)
        saved = _grab_fresh_mic_stills(base + "_mic", t_done)
        _trial_capture["base"] = base
        tip = cur_tip()
        meta = {
            "trial": trial, "timestamp": t_done, "timestamp_iso": datetime.datetime.fromtimestamp(t_done).isoformat(),
            "stills": saved, "tip_base_mm": tip, "plunge_goal_base_mm": plunge_goal,
            "internal_rcm_base_mm": S.internal_rcm, "rcm_target_base_mm": _vision_rcm.get("p"),
            "rcm_source": S.rcm_source, "mic_rcm_mode": MIC_RCM_MODE, "mic_target_mark": MIC_TARGET_MARK,
            "mic_rcm_angle_deg": MIC_RCM_ANGLE_DEG,
            "phantom_normal_base": (PHANTOM_NORMAL or {}).get("normal"),
            "insert_angle_deg": INSERT_ANGLE_DEG, "plunge_depth_mm": PLUNGE_DEPTH_MM,
            "mic_lock_rec": _mic_prev.get("rec"), "mic_geo_freeze": _mic_geo_freeze.get("geo"),
            "last_correspondence": dict(_last_correspondence),
            "origin_correction_mm": ORIGIN_CORRECTION,
        }
        with open(base + ".json", "w") as fh:
            json.dump(_jsonable(meta), fh, indent=1)
        S.msg = ("trial %d: saved %d/%d post-PLUNGE microscope stills -> %s" %
                 (trial, len(saved), TRIAL_STILLS_N, os.path.basename(base)))
        print("  " + S.msg)
        if not saved:
            print("  !! no fresh microscope still arrived within %.0fs -- is mic_live_capture running?" %
                  TRIAL_STILLS_TIMEOUT_S)

    def estimate_phantom_normal_from_ring(rcm_base):
        """Phantom normal = plane of the red ring, averaged over NORMAL_RING_FRAMES
        fresh frames. Same return dict as estimate_phantom_normal_from_frame()."""
        if not handeye_ok:
            return {"ok": False, "fail_reason": "hand-eye missing"}
        normals, inl_counts = [], []
        last_center = _ring_live["last_center"]
        for _ in range(NORMAL_RING_FRAMES):
            try:
                fr = align.process(pipe.wait_for_frames(timeout_ms=1000))
            except Exception:
                continue
            cf, df = fr.get_color_frame(), fr.get_depth_frame()
            if not cf or not df:
                continue
            img = np.asanyarray(cf.get_data())
            cx, cy, r_px, _, _ = detect_ring(img, last_center)
            if cx is None:
                continue
            last_center = (cx, cy)
            dmm = np.asanyarray(df.get_data()).astype(np.float64) * depth_scale * 1000.0
            coef, n_inl, _ = ring_plane_fit(img, dmm, cx, cy, r_px)
            if coef is None:
                continue
            normals.append(ring_plane_normal_cam(coef, cx, cy, intr.fx, intr.fy, intr.ppx, intr.ppy))
            inl_counts.append(n_inl)
        need = max(3, NORMAL_RING_FRAMES // 2)
        if len(normals) < need:
            return {"ok": False, "n_points": 0, "n_red_dropped": 0,
                    "fail_reason": "ring plane fit succeeded on only %d/%d frames (need %d) -- "
                                    "is the red ring fully visible to the D405?" %
                                    (len(normals), NORMAL_RING_FRAMES, need)}
        N = np.array(normals)
        normal_cam = N.mean(axis=0)
        normal_cam /= np.linalg.norm(normal_cam)
        spread_deg = float(np.degrees(np.arccos(np.clip(N.dot(normal_cam), -1, 1))).max())
        if spread_deg > NORMAL_RING_MAX_SPREAD_DEG:
            return {"ok": False, "n_points": int(np.sum(inl_counts)), "n_red_dropped": 0,
                    "fail_reason": "ring-plane normal unstable: max %.1fdeg from mean over %d frames "
                                    "(limit %.1f) -- keep the phantom/tool still and retry" %
                                    (spread_deg, len(normals), NORMAL_RING_MAX_SPREAD_DEG)}
        normal_base = R_he.dot(normal_cam)
        normal_base /= np.linalg.norm(normal_base)
        print("  [normal, ring plane] %d frames, max spread %.1fdeg, tilt from base +Z %.1fdeg" %
              (len(normals), spread_deg, np.degrees(np.arccos(np.clip(normal_base[2], -1, 1)))))
        return {"ok": True, "normal_base": normal_base, "normal_cam": normal_cam,
                "point_base": rcm_base.copy(), "n_points": int(np.mean(inl_counts)),
                "n_red_dropped": 0, "plane_rms_mm": float("nan"), "spread_deg": spread_deg,
                "centroid_disagree_mm": 0.0, "centroid_lateral_mm": 0.0, "centroid_depth_mm": 0.0,
                "fail_reason": None}

    def _draw_scene(view, k=1.0):
        """Tip / RCM / home / shaft / insertion axis / phantom ring, scaled by k."""
        def K(p):
            return None if p is None else (int(round(p[0] * k)), int(round(p[1] * k)))
        lw = max(1, int(round(k)))
        ts = 0.38 * (1.0 + (k - 1.0) * 0.8)
        if handeye_ok:
            tip_px = K(base_to_pixel(cur_tip(), R_he, t_he, intr)) if cur_tip() is not None else None
            rcm_px = K(base_to_pixel(cur_rcm(), R_he, t_he, intr)) if rcm_ok() else None
            home_px = K(base_to_pixel(S.home, R_he, t_he, intr)) if S.home is not None else None
            hover_px = K(base_to_pixel(S.hover_goal, R_he, t_he, intr)) if S.hover_goal is not None else None
            if tip_px is not None and rcm_px is not None:
                cv2.line(view, tip_px, rcm_px, (0, 0, 0), lw + 2, cv2.LINE_AA)
                cv2.line(view, tip_px, rcm_px, COLOR_SHAFT, lw, cv2.LINE_AA)
            if PHANTOM_NORMAL is not None and rcm_ok():
                tdir = desired_insertion_dir()
                if tdir is not None:
                    rcm_pt = cur_rcm()
                    p1 = K(base_to_pixel(rcm_pt - tdir * 15.0, R_he, t_he, intr))
                    p2 = K(base_to_pixel(rcm_pt + tdir * 6.0, R_he, t_he, intr))
                    if p1 is not None and p2 is not None:
                        cv2.line(view, p1, p2, COLOR_TARGET, lw, cv2.LINE_AA)
            rr = lambda r: max(3, int(round(r * (1.0 + (k - 1.0) * 0.7))))
            draw_dot(view, home_px, COLOR_HOME, "HOME", radius=rr(4), label_left=True, tscale=ts)
            draw_dot(view, hover_px, COLOR_HOVER, "HOVER", radius=rr(4), tscale=ts)
            draw_dot(view, rcm_px, COLOR_RCM, "RCM", radius=rr(6), tscale=ts)
            draw_dot(view, tip_px, COLOR_TIP, "TIP", radius=rr(4), tscale=ts)
        if _ring_live["cx"] is not None and _ring_live["r_px"] is not None:
            ring_col = HUD_OK if _ring_live["locked"] else HUD_WARN
            _rc = (int(round(_ring_live["cx"] * k)), int(round(_ring_live["cy"] * k)))
            _rr = int(round(_ring_live["r_px"] * k))
            cv2.circle(view, _rc, _rr, (0, 0, 0), lw + 2, cv2.LINE_AA)
            cv2.circle(view, _rc, _rr, ring_col, lw, cv2.LINE_AA)

    def draw(view):
        _draw_scene(view, 1.0)
        if _aruco_live["found"] and _aruco_live.get("corners") is not None:
            pts = _aruco_live["corners"].astype(int).reshape(-1, 1, 2)
            cv2.polylines(view, [pts], True, HUD_OK, 2, cv2.LINE_AA)
        H_, W_ = view.shape[:2]

        # ---- bottom-left status table: one coloured dot + short value per subsystem
        rows = []
        if _ring_live["locked"]:
            rows.append(("RING", "locked  +/-%.2f mm" % (_ring_live["jitter"] or 0.0), HUD_OK))
        elif _ring_live["det"]:
            rows.append(("RING", "settling %d/%d" % (_ring_live["n"], RING_SMOOTH_N), HUD_WARN))
        else:
            rows.append(("RING", "not visible", HUD_BAD))
        if MIC_RCM_MODE == "circle":
            mic_locked = _mic_target_live["locked"]
            mic_msg = _mic_target_live["status_msg"] or ""
        else:
            mic_locked = _mic_mark_live["locked"].get(MIC_TARGET_MARK, False)
            mic_msg = _mic_mark_live["status_msg"] or ""
        _mic_tag_s = ("circle@%.0f" % MIC_RCM_ANGLE_DEG) if MIC_RCM_MODE == "circle" else ("mark%d" % MIC_TARGET_MARK)
        if mic_locked:
            rows.append(("MIC", "locked  %s" % _mic_tag_s, HUD_OK))
        else:
            _m = re.search(r"(\d+/\d+) samples", mic_msg)
            if _m:
                rows.append(("MIC", "settling %s" % _m.group(1), HUD_WARN))
            elif "occlu" in mic_msg or "unreliable" in mic_msg:
                rows.append(("MIC", "occluded / no radius", HUD_BAD))
            else:
                rows.append(("MIC", "waiting", HUD_WARN))
        rows.append(("NORMAL", "set" if PHANTOM_NORMAL is not None else "not set  (n)",
                     HUD_OK if PHANTOM_NORMAL is not None else HUD_WARN))
        if _aruco_live["found"]:
            rows.append(("ARUCO", "%s visible" % (_aruco_live["label"] or ""), HUD_OK))
        if _trial_state["idx"] > 0:
            if len(_trial_state["home_history"]) >= 2:
                _cur_h = _trial_state["home_history"][-1][1]
                _min_d = min(float(np.linalg.norm(_cur_h - h))
                             for i, h in _trial_state["home_history"][:-1])
                rows.append(("TRIAL", "#%d   home +%.1f mm" % (_trial_state["idx"], _min_d),
                             HUD_OK if _min_d >= HOME_DIVERSITY_MIN_MM else HUD_WARN))
            else:
                rows.append(("TRIAL", "#%d   first" % _trial_state["idx"], HUD_OK))
        res_line = None
        if _last_correspondence["t"] is not None:
            dc = _last_correspondence["disc_corr_mm"]
            dr = _last_correspondence["disc_raw_mm"]
            headline = dc if dc is not None else dr
            res_col = HUD_OK if headline < 1.0 else (HUD_WARN if headline < 3.0 else HUD_BAD)
            res_line = ("MIC<->D405  %.2f mm" % headline,
                        ("raw %.2f  |  %.0fs ago" % (dr, time.time() - _last_correspondence["t"]))
                        + ("" if dc is not None else "  |  uncorrected"), res_col)
        bottom_bar_h = 44
        row_h = 17
        panel_h = len(rows) * row_h + 8 + (24 if res_line else 0)
        py1 = H_ - bottom_bar_h - 6
        py0 = py1 - panel_h
        hud_panel(view, 4, py0, 236, py1, 0.55)
        y = py0 + 16
        for lab, val, col in rows:
            hud_status_row(view, y, lab, val, col, x=10, max_w=222)
            y += row_h
        if res_line:
            hud_text(view, res_line[0], (12, y + 4), 0.52, res_line[2], 1, max_w=220)
            hud_text(view, res_line[1], (12, y + 18), 0.34, HUD_DIM, max_w=220)

        # ---- top bar: state / RCM / angle, then a dim second line
        hud_panel(view, 0, 0, W_, 40, 0.62)
        x = 10 + hud_text(view, S.state, (10, 18), 0.55, HUD_ACCENT, 1) + 16
        if rcm_ok():
            rcm_col = HUD_OK if S.rcm_source == "vision" else HUD_WARN
            rcm_val = S.rcm_source or "manual"
        else:
            rcm_col, rcm_val = HUD_BAD, "not latched"
        cv2.circle(view, (x + 4, 14), 4, rcm_col, -1, cv2.LINE_AA)
        x += 14 + hud_text(view, "RCM %s" % rcm_val, (x + 12, 18), 0.45, HUD_TXT) + 18
        hud_text(view, "angle %.0f deg" % INSERT_ANGLE_DEG, (x, 18), 0.45, HUD_TXT)
        # pt.17 (2026-08-23): dedicated NORMAL-mode line -- az=... alone (buried
        # in the STATE line above) was too easy to miss/misread on a screenshot
        # or recording as "which mode is this actually running", exactly what
        # was asked after the first dual-axis run. Distinct color per mode
        # (green=dual-axis/pt.16 default, orange=legacy) so it reads correctly
        # even from a paused video frame or PPT screenshot with no audio/context.
        if INSERT_AZIMUTH_DEG is not None:
            mode_txt, mode_col = "dual-axis  az %.0f" % INSERT_AZIMUTH_DEG, HUD_DIM
        else:
            mode_txt, mode_col = "single-axis LEGACY (not for reporting)", HUD_WARN
        line2 = "lin %.1f mm/s   ang %.3f rad/s   |   " % (SPEED, ASPEED)
        lx = 10 + hud_text(view, line2, (10, 34), 0.36, HUD_DIM)
        hud_text(view, mode_txt + ("   |   biplanar" if BIPLANAR_MODE else "   |   single-angle"),
                 (lx, 34), 0.36, mode_col)
        hint = "Tab: keys"
        hud_text(view, hint, (W_ - 10 - hud_tw(hint, 0.36), 34),
                 0.36, HUD_DIM)
        if S.rec:
            cv2.circle(view, (W_ - 52, 13), 5, (0, 0, 255), -1, cv2.LINE_AA)
            hud_text(view, "REC", (W_ - 42, 18), 0.45, (0, 0, 255), 1)

        # ---- bottom bar: last message (+ vision-check line)
        hud_panel(view, 0, H_ - bottom_bar_h, W_, H_, 0.62)
        if S.msg:
            hud_text(view, S.msg, (10, H_ - bottom_bar_h + 16), 0.42, (80, 230, 255), 1, max_w=W_ - 20)
        if S.last_vis_check_msg:
            hud_text(view, S.last_vis_check_msg, (10, H_ - 10), 0.34, COLOR_VISCHK, 1, max_w=W_ - 230)
        # 2026-09-21: microscope picture-in-picture -- shows the NEWEST still
        # dropped in MIC_LIVE_STILLS_DIR (auto-refreshes as soon as a new file
        # lands there, no key press needed), with the classical-CV landmark
        # fit drawn on it and the mark 'y' would lock highlighted in green.
        # UPDATE 2026-09-22: DeckLink capture card + gscam_decklink.launch +
        # mic_live_capture_ros.py are now live (1920x1080@59.94 SDI feed from
        # the Leica CCU), so this folder is refreshed continuously, not by
        # manual file transfer anymore. Staleness is still fully enforced --
        # ok_col below still goes red/orange if age_s > MIC_LIVE_MAX_AGE_S or
        # the frame is occluded -- only the on-screen label was simplified.
        # Press 'y' to commit this frame's mark as the RCM point.
        try:
            preview = get_mic_preview()
        except Exception as _e:
            preview = None
            hud_text(view, "Microscope preview error: %s" % _e, (view.shape[1] - 268, 70), 0.34,
                     HUD_BAD, max_w=260)
        if preview is not None:
            thumb, prec, ppath, pmtime = preview
            age_s = time.time() - pmtime
            fresh = age_s <= MIC_LIVE_MAX_AGE_S
            ok_col = (0, 255, 0) if (fresh and prec["ok"] and not prec["occluded"]) else \
                     ((0, 165, 255) if fresh else (0, 0, 255))
            th, tw = thumb.shape[:2]
            x0 = view.shape[1] - tw - 8
            HDR, FTR = 20, 20
            yh = 48                      # header top
            y0 = yh + HDR                # thumbnail top
            if fresh and prec["ok"] and not prec["occluded"]:
                mic_state = "Live"
            elif not fresh:
                mic_state = "Stale"
            else:
                mic_state = "Occluded"
            st_col = HUD_OK if mic_state == "Live" else (HUD_WARN if mic_state == "Occluded" else HUD_BAD)
            view[y0:y0 + th, x0:x0 + tw] = thumb
            for (ya, yb) in ((yh, y0), (y0 + th, y0 + th + FTR)):
                roi = view[ya:yb, x0:x0 + tw]
                roi[:] = (roi.astype(np.float32) * 0.22 + 20).astype(np.uint8)
            hud_text(view, "Leica Microscope", (x0 + 8, yh + 14), 0.38, HUD_TXT, bold=True)
            sw = hud_tw(mic_state, 0.36)
            hud_text(view, mic_state, (x0 + tw - 8 - sw, yh + 14), 0.36, st_col)
            cv2.circle(view, (x0 + tw - 8 - sw - 8, yh + 10), 3, st_col, -1, cv2.LINE_AA)
            # legend
            ly = y0 + th + 14
            lx = x0 + 8
            _lk = _mic_target_live["locked"] if MIC_RCM_MODE == "circle" else \
                _mic_mark_live["locked"].get(MIC_TARGET_MARK, False)
            for name, col, kind in (("Limbus", MIC_C_LIMBUS, "line"),
                                    ("Marker ring" if MIC_RCM_MODE == "circle" else "Outer ring",
                                     MIC_C_MARKRING, "line"),
                                    ("RCM", MIC_C_LOCK if _lk else MIC_C_SEEK, "dot")):
                if kind == "line":
                    cv2.line(view, (lx, ly - 4), (lx + 12, ly - 4), col, 2, cv2.LINE_AA)
                else:
                    cv2.circle(view, (lx + 5, ly - 4), 4, col, 1, cv2.LINE_AA)
                    cv2.circle(view, (lx + 5, ly - 4), 1, col, -1, cv2.LINE_AA)
                lx += 17 + hud_text(view, name, (lx + 17, ly), 0.34, HUD_DIM) + 8
            cv2.rectangle(view, (x0 - 1, yh - 1), (x0 + tw, y0 + th + FTR), (95, 95, 95), 1)
        else:
            # 2026-09-21: explicit placeholder -- MIC_LIVE_STILLS_DIR being
            # empty must NEVER look identical to "feature not present"; a
            # silently-missing panel here is exactly what caused confusion
            # the first time this shipped. Always draw SOMETHING in this
            # corner so its absence is self-explanatory from the video alone.
            tw, th = 260, 146
            x0 = view.shape[1] - tw - 8
            y0 = 48
            hud_panel(view, x0, y0, x0 + tw, y0 + th + 40, 0.7)
            cv2.rectangle(view, (x0 - 1, y0 - 1), (x0 + tw, y0 + th + 40), (95, 95, 95), 1)
            hud_text(view, "Leica Microscope", (x0 + 8, y0 + 14), 0.38, HUD_TXT, bold=True)
            hud_text(view, "No frames received", (x0 + 8, y0 + 80), 0.40, HUD_BAD)
            hud_text(view, "check " + os.path.basename(MIC_LIVE_STILLS_DIR) + "/", (x0 + 8, y0 + 100),
                     0.34, HUD_DIM, max_w=tw - 16)
        if _hud["help"]:
            draw_help(view, 1.0)
        return view

    # ------------------------------------------------------------------
    # Demo layout (1280x720): clean multi-panel composition for videos.
    # The dense diagnostic HUD above is kept as the debug view (key V).
    # ------------------------------------------------------------------
    PRES_W, PRES_H = 1280, 720
    _gamma_lut = np.array([min(255, int(((i / 255.0) ** 0.78) * 255.0 + 0.5)) for i in range(256)], np.uint8)
    PRES_BG, PRES_CARD, PRES_LINE = (26, 24, 24), (36, 34, 34), (66, 64, 64)
    PRES_STEPS = ["Setup", "Approach", "Orient", "Tunnel", "Plunge", "Retract"]
    PRES_DONE_N = {"HOVER": 1, "ORIENTED": 2, "INSERTED": 3, "TUNNELED": 3, "PLUNGED": 4, "RETRACTED": 5}

    def draw_present(img):
        CW, CH = PRES_W, PRES_H
        c = np.empty((CH, CW, 3), np.uint8)
        c[:] = PRES_BG
        MX, MY, MW = 780, 92, 472
        MH = int(round(MW * 9 / 16.0))
        MIC_THUMB_W["w"] = MW

        # ---- title bar
        hud_text(c, DEMO_TITLE, (28, 27), 0.66, (245, 245, 245), bold=True)
        if DEMO_SUBTITLE:
            hud_text(c, DEMO_SUBTITLE, (28, 46), 0.42, HUD_DIM)
        rx = CW - 28
        if S.rec:
            rx -= hud_tw("REC", 0.46, True)
            hud_text(c, "REC", (rx, 34), 0.46, (80, 80, 255), bold=True)
            cv2.circle(c, (rx - 12, 27), 5, (80, 80, 255), -1, cv2.LINE_AA)
            rx -= 36
        if _trial_state["idx"] > 0:
            tl = "Trial %d" % _trial_state["idx"]
            rx -= hud_tw(tl, 0.5)
            hud_text(c, tl, (rx, 34), 0.5, HUD_TXT)
        cv2.line(c, (0, 52), (CW, 52), (48, 46, 46), 1)

        # ---- left: depth camera with geometry only
        PX, PY, PW, PH = 28, 92, 720, 540
        bg = cv2.LUT(cv2.resize(img, (PW, PH), interpolation=cv2.INTER_CUBIC), _gamma_lut)
        _draw_scene(bg, PW / 640.0)
        c[PY:PY + PH, PX:PX + PW] = bg
        cv2.rectangle(c, (PX - 1, PY - 1), (PX + PW, PY + PH), PRES_LINE, 1)
        hud_text(c, "RealSense Depth Camera", (PX, 82), 0.52, HUD_TXT, bold=True)
        ang = "insertion angle %.0f\u00b0" % INSERT_ANGLE_DEG
        hud_text(c, ang, (PX + PW - hud_tw(ang, 0.44), 82), 0.44, HUD_DIM)

        # ---- right: microscope
        try:
            prev = get_mic_preview()
        except Exception:
            prev = None
        hud_text(c, "Leica Microscope", (MX, 82), 0.52, HUD_TXT, bold=True)
        if prev is not None:
            thumb, prec, ppath, pmtime = prev
            fresh = (time.time() - pmtime) <= MIC_LIVE_MAX_AGE_S
            if thumb.shape[1] != MW or thumb.shape[0] != MH:
                thumb = cv2.resize(thumb, (MW, MH), interpolation=cv2.INTER_AREA)
            c[MY:MY + MH, MX:MX + MW] = thumb
            if ppath != _mic_occ_run["path"]:
                _mic_occ_run["path"] = ppath
                _mic_occ_run["n"] = (_mic_occ_run["n"] + 1) if (not prec["ok"] or prec["occluded"]) else 0
            if not fresh:
                st, st_col = "No signal", HUD_BAD
            elif _mic_occ_run["n"] < MIC_OCC_SHOW_N:
                st, st_col = "Live", HUD_OK
            else:
                st, st_col = "Occluded", HUD_WARN
            sw = hud_tw(st, 0.44)
            hud_text(c, st, (MX + MW - sw, 82), 0.44, st_col)
            cv2.circle(c, (MX + MW - sw - 10, 77), 4, st_col, -1, cv2.LINE_AA)
        else:
            c[MY:MY + MH, MX:MX + MW] = PRES_CARD
            hud_text(c, "No microscope frames", (MX + 20, MY + MH // 2), 0.5, HUD_BAD)
        cv2.rectangle(c, (MX - 1, MY - 1), (MX + MW, MY + MH), PRES_LINE, 1)
        ly = MY + MH

        # ---- right: status card
        cy0, cy1 = ly + 16, 632
        c[cy0:cy1, MX:MX + MW] = PRES_CARD
        cv2.rectangle(c, (MX, cy0), (MX + MW - 1, cy1 - 1), PRES_LINE, 1)
        hud_text(c, "System status", (MX + 18, cy0 + 26), 0.46, HUD_TXT, bold=True)
        if _ring_live["locked"]:
            r_ = ("Locked  \u00b1%.2f mm" % (_ring_live["jitter"] or 0.0), HUD_OK)
        elif _ring_live["det"]:
            r_ = ("Settling  %d/%d" % (_ring_live["n"], RING_SMOOTH_N), HUD_WARN)
        else:
            r_ = ("Not visible", HUD_BAD)
        if MIC_RCM_MODE == "circle":
            m_lock, m_msg = _mic_target_live["locked"], (_mic_target_live["status_msg"] or "")
        else:
            m_lock, m_msg = _mic_mark_live["locked"].get(MIC_TARGET_MARK, False), (_mic_mark_live["status_msg"] or "")
        if m_lock:
            m_ = ("Locked", HUD_OK)
        else:
            _mm = re.search(r"(\d+/\d+) samples", m_msg)
            if _mm:
                m_ = ("Settling  %s" % _mm.group(1), HUD_WARN)
            elif "occlu" in m_msg or "unreliable" in m_msg:
                m_ = ("Occluded", HUD_BAD)
            else:
                m_ = ("Waiting", HUD_WARN)
        n_ = ("Estimated", HUD_OK) if PHANTOM_NORMAL is not None else ("Not set", HUD_WARN)
        probs = _calib_fresh_cache["probs"]
        k_ = ("Update needed", HUD_WARN) if probs else ("Valid", HUD_OK)
        for i, (lab, (val, col)) in enumerate((("Ring detection", r_), ("Leica RCM", m_),
                                               ("Surface normal", n_), ("Calibration", k_))):
            yy = cy0 + 60 + 27 * i
            cv2.circle(c, (MX + 24, yy - 5), 4, col, -1, cv2.LINE_AA)
            hud_text(c, lab, (MX + 38, yy), 0.46, HUD_DIM)
            hud_text(c, val, (MX + 190, yy), 0.48, HUD_TXT, max_w=MW - 210)
        cv2.line(c, (MX + 18, cy0 + 164), (MX + MW - 18, cy0 + 164), PRES_LINE, 1)
        hud_text(c, "Agreement: Leica vs. RealSense", (MX + 18, cy0 + 190), 0.42, HUD_DIM)
        if _last_correspondence["t"] is not None:
            dc, dr = _last_correspondence["disc_corr_mm"], _last_correspondence["disc_raw_mm"]
            hl = dc if dc is not None else dr
            hcol = HUD_OK if hl < 1.0 else (HUD_WARN if hl < 3.0 else HUD_BAD)
            nw = hud_text(c, "%.2f" % hl, (MX + 18, cy1 - 18), 1.25, hcol, bold=True)
            hud_text(c, "mm", (MX + 18 + nw + 8, cy1 - 18), 0.55, HUD_DIM)
            sub = "raw %.2f mm   \u00b7   %.0f s ago" % (dr, time.time() - _last_correspondence["t"])
            hud_text(c, sub, (MX + MW - 18 - hud_tw(sub, 0.4), cy1 - 18), 0.4, HUD_DIM)
        else:
            hud_text(c, "\u2014", (MX + 18, cy1 - 18), 1.25, HUD_DIM, bold=True)
            hud_text(c, "press ` to measure", (MX + 70, cy1 - 18), 0.42, HUD_DIM)

        # ---- bottom: phase stepper
        setup_ok = _ring_live["locked"] and m_lock and PHANTOM_NORMAL is not None
        n_done = PRES_DONE_N.get(S.state, 0)
        done = [setup_ok or n_done > 0] + [n_done >= i for i in range(1, 6)]
        active = next((i for i, d in enumerate(done) if not d), None)
        x_a, x_b = 28 + 60, CW - 28 - 60
        step = (x_b - x_a) / float(len(PRES_STEPS) - 1)
        sy = 658
        for i in range(len(PRES_STEPS) - 1):
            xa, xb = int(x_a + step * i + 14), int(x_a + step * (i + 1) - 14)
            cv2.line(c, (xa, sy), (xb, sy), (110, 190, 120) if done[i] and done[i + 1] else (70, 68, 68), 2, cv2.LINE_AA)
        for i, name in enumerate(PRES_STEPS):
            cx = int(x_a + step * i)
            if done[i]:
                cv2.circle(c, (cx, sy), 11, (110, 190, 120), -1, cv2.LINE_AA)
                cv2.polylines(c, [np.array([(cx - 5, sy), (cx - 1, sy + 4), (cx + 5, sy - 4)], np.int32)],
                              False, (24, 24, 24), 2, cv2.LINE_AA)
                tcol = HUD_TXT
            elif i == active:
                cv2.circle(c, (cx, sy), 11, HUD_ACCENT, 2, cv2.LINE_AA)
                cv2.circle(c, (cx, sy), 4, HUD_ACCENT, -1, cv2.LINE_AA)
                tcol = HUD_ACCENT
            else:
                cv2.circle(c, (cx, sy), 11, (88, 86, 86), 1, cv2.LINE_AA)
                tcol = (120, 120, 120)
            nw_ = hud_tw(name, 0.46, i == active)
            hud_text(c, name, (cx - nw_ // 2, sy + 31), 0.46, tcol, bold=(i == active))
        # operator message (one dim line)
        if S.msg:
            hud_text(c, S.msg, (28, CH - 7), 0.38, (150, 150, 150), max_w=CW - 56 - 90)
        hud_text(c, "Tab keys", (CW - 28 - hud_tw("Tab keys", 0.38), CH - 8), 0.38, (110, 110, 110))
        if _hud["help"]:
            draw_help(c, 1.6)
        return c

    win = "HOME -> RCM INSERTION [v2 precision]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720) if _present["on"] else cv2.resizeWindow(win, 960, 720)

    _calib = {"armed": False, "u": None, "v": None}

    def _on_mouse_calib(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and _calib["armed"]:
            _calib["u"], _calib["v"] = x, y

    cv2.setMouseCallback(win, _on_mouse_calib)

    def calibrate_origin_from_click():
        """'c': measure the constant base-frame offset between the encoder
        pipeline (/FrameEE -> cur_tip()) and reality, using the camera+depth
        ground truth -- same instrument that's already trusted for the
        independent visual verification work. Click the ACTUAL needle tip
        in the window; ENTER accepts, ESC cancels. Does not move the robot.
        """
        if not handeye_ok:
            S.msg = "no hand-eye loaded -- cannot calibrate origin correction"; return
        if cur_tip() is None:
            S.msg = "no FrameEE yet -- cannot calibrate origin correction"; return
        ee_p_at_click = _ee["p"].copy()
        ee_q_at_click = _ee["q"].copy()
        cur_tip_at_click = cur_tip().copy()
        _calib["armed"] = True; _calib["u"] = None; _calib["v"] = None
        print("  [ORIGIN CALIBRATION] click the ACTUAL visible needle tip in the "
              "window, ENTER=accept ESC=cancel")
        while True:
            frames_c = pipe.wait_for_frames(timeout_ms=2000)
            frames_c = align.process(frames_c)
            color_fc = frames_c.get_color_frame()
            depth_fc = frames_c.get_depth_frame()
            if not color_fc or not depth_fc:
                continue
            img_c = np.asanyarray(color_fc.get_data()).copy()
            cv2.rectangle(img_c, (0, 0), (640, 24), (0, 0, 0), -1)
            cv2.putText(img_c, "ORIGIN CALIB: click ACTUAL needle tip  ENTER=accept ESC=cancel",
                        (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            if _calib["u"] is not None:
                cv2.drawMarker(img_c, (_calib["u"], _calib["v"]), (0, 0, 255),
                                cv2.MARKER_CROSS, 18, 2)
            cv2.imshow(win, img_c)
            kk = cv2.waitKey(20) & 0xFF
            if kk == 27:
                S.msg = "origin calibration cancelled"; _calib["armed"] = False; return
            if kk in (13, 10) and _calib["u"] is not None:
                u, v = _calib["u"], _calib["v"]
                # Thin-needle stereo depth is unreliable, so only accept a
                # measured depth within ORIGIN_CLICK_DZ_MAX_M of where the
                # current model already puts the tip; otherwise use the model
                # depth and correct only the in-image (2D) offset.
                z_pred = float(R_he.T.dot(cur_tip_at_click - t_he)[2]) / 1000.0
                z_m, z_win, z_n, z_far = needle_depth_m_at_click(
                    depth_fc, u, v,
                    z_min=max(ORIGIN_CLICK_Z_MIN_M, z_pred - ORIGIN_CLICK_DZ_MAX_M),
                    z_max=min(ORIGIN_CLICK_Z_MAX_M, z_pred + ORIGIN_CLICK_DZ_MAX_M))
                depth_mode = "measured"
                if z_m is None:
                    z_m, depth_mode = z_pred, "model (2D-only)"
                if not (ORIGIN_CLICK_Z_MIN_M <= z_m <= ORIGIN_CLICK_Z_MAX_M):
                    S.msg = ("ORIGIN CALIB REJECTED: predicted tip depth %.0fmm outside "
                             "%.0f-%.0fmm" % (z_m * 1000.0, ORIGIN_CLICK_Z_MIN_M * 1000.0,
                                              ORIGIN_CLICK_Z_MAX_M * 1000.0))
                    print("  " + S.msg)
                    _calib["armed"] = False; return
                cam_pt_mm = np.array(deproject_mm(intr, u, v, z_m), np.float64)
                true_tip_base = R_he.dot(cam_pt_mm) + t_he
                # 2026-10-06: FOUND A SECOND, BIGGER BUG HERE -- cur_tip_at_click
                # was computed from cur_tip(), which already has the OLD
                # ORIGIN_CORRECTION baked in (see _cb_ee: _ee["p"] = raw +
                # ORIGIN_CORRECTION). So (true_tip_base - cur_tip_at_click) is
                # NOT the new absolute correction -- it's
                # (true_new_correction - OLD_correction). Overwriting
                # ORIGIN_CORRECTION with THAT value (instead of adding it to
                # the old one) makes the result wrong by exactly however much
                # the old correction already was, every time 'c' is pressed
                # with a nonzero correction already in effect (true from the
                # very first press today, since we loaded yesterday's nonzero
                # correction at startup). Each repeated 'c' press was
                # compounding this, not converging -- exactly the "pressed c
                # several times, still misaligned" symptom. Fix: ACCUMULATE
                # (old + delta), matching what "measure a correction, apply
                # it" is actually supposed to mean.
                correction = ORIGIN_CORRECTION + (true_tip_base - cur_tip_at_click)
                mag = float(np.linalg.norm(correction))
                if mag > ORIGIN_CORRECTION_MAX_MM:
                    S.msg = ("ORIGIN CALIB REJECTED: |corr|=%.1fmm > %.1fmm -- this is "
                             "the far-Z / background-depth failure (green dot would "
                             "align here then barely track). depth=%.0fmm win=%d n=%d. "
                             "SHIFT+C keeps zero; do not accept this click." % (
                                 mag, ORIGIN_CORRECTION_MAX_MM, z_m * 1000.0, z_win, z_n))
                    print("  " + S.msg)
                    _calib["armed"] = False; return
                ORIGIN_CORRECTION[:] = correction
                np.savez(ORIGIN_CORRECTION_NPZ, correction=correction,
                         clicked_px=[u, v], depth_m=z_m, ee_p_at_click=ee_p_at_click,
                         ee_q_at_click=ee_q_at_click, cur_tip_at_click=cur_tip_at_click,
                         timestamp=time.time())
                S.msg = ("ORIGIN CORRECTION applied: [%.2f %.2f %.2f]mm "
                          "(|corr|=%.1fmm, Z=%.0fmm %s) -- press 'h' to re-latch HOME" % (
                              correction[0], correction[1], correction[2], mag,
                              z_m * 1000.0, depth_mode))
                print("  " + S.msg + ("  (saved to %s)" % ORIGIN_CORRECTION_NPZ))
                _calib["armed"] = False
                return

    def clear_origin_correction():
        """SHIFT+C: drop a bad ORIGIN_CORRECTION back to zero without a restart."""
        ORIGIN_CORRECTION[:] = 0.0
        try:
            if os.path.isfile(ORIGIN_CORRECTION_NPZ):
                bak = ORIGIN_CORRECTION_NPZ + ".cleared_%d" % int(time.time())
                os.rename(ORIGIN_CORRECTION_NPZ, bak)
                print("  origin_correction.npz moved aside -> %s" % bak)
        except Exception as e:
            print("  (could not move origin_correction.npz aside: %s)" % e)
        S.msg = "ORIGIN CORRECTION cleared to [0 0 0] -- green TIP is now raw kinematics+hand-eye"
        print("  " + S.msg)

    def log_mic_calib_touch(label):
        """'1'-'8': log the robot's CURRENT base-frame tip position
        (cur_tip(), same pipeline the whole controller trusts) under a named
        label, for the offline microscope-image -> robot-base-frame
        extrinsic calibration (see /home/eyebrp/Matthew/microscope_autonomy/
        mic_extrinsic_calib.py). Does NOT move the robot or touch the
        camera -- physically drive the tip to rest on the named phantom
        landmark FIRST (e.g. one of the two ink entry marks, or another
        sharp, unambiguous feature you can also click in the microscope
        still afterward), THEN press the key. Appends one line to
        MIC_CALIB_LOG; safe to press repeatedly (e.g. average several
        touches of the same point -- the fitting script averages by label).

        2026-09-24: use ALL 8 keys, spread across the full field of view
        (not clustered near the 2 ink marks) -- 4 points exactly match the
        6-DOF affine model's ability to memorize them (only 2 residual DOF
        left), which makes the in-sample fit residual look good almost by
        construction, not because the map is accurate away from those 4
        points. 8 points gives mic_extrinsic_calib.py --loo something
        real to hold out and report an honest, non-self-referential
        accuracy number. See MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md.

        2026-10-01 (pt.2): ALSO copies whatever is currently the freshest
        frame in MIC_LIVE_STILLS_DIR into MIC_CALIB_TOUCH_STILLS_DIR as
        "<label>.jpg" -- the tool tip is still resting on the just-touched
        point in that frame, so this snapshot visibly marks its own pixel
        location for click_calib_points_per_touch.py, instead of relying on
        the operator remembering/re-identifying 8 touch locations on one
        later, separate still (see MIC_CALIB_TOUCH_STILLS_DIR's own comment
        for the full rationale). Best-effort: a missing/stale live still
        does NOT block logging the touch itself (the 3D measurement is the
        thing that must never be lost), it only means that label will have
        no auto-captured still and needs a manual fallback.
        """
        if cur_tip() is None:
            S.msg = "no FrameEE yet -- cannot log calibration touch"; return
        rec = {
            "label": label,
            "cur_tip_base_mm": cur_tip().tolist(),
            "ee_p_base_mm": _ee["p"].tolist(),
            "ee_q": _ee["q"].tolist(),
            "timestamp": time.time(),
            "timestamp_iso": datetime.datetime.now().isoformat(),
        }
        still_note = "no live still captured"
        try:
            imgs = [os.path.join(MIC_LIVE_STILLS_DIR, f) for f in os.listdir(MIC_LIVE_STILLS_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if imgs:
                latest = max(imgs, key=os.path.getmtime)
                age_s = time.time() - os.path.getmtime(latest)
                os.makedirs(MIC_CALIB_TOUCH_STILLS_DIR, exist_ok=True)
                dest = os.path.join(MIC_CALIB_TOUCH_STILLS_DIR, label + ".jpg")
                shutil.copyfile(latest, dest)
                rec["still_image"] = dest
                rec["still_age_s_at_touch"] = age_s
                still_note = "still -> %s (%.1fs old)" % (os.path.basename(dest), age_s)
                if age_s > MIC_LIVE_MAX_AGE_S:
                    still_note += "  WARNING: stale, tip may have moved since this frame"
        except OSError as error:
            still_note = "still capture failed: %s" % error
        with open(MIC_CALIB_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        t = rec["cur_tip_base_mm"]
        S.msg = ("MIC CALIB TOUCH '%s' logged: tip=[%.2f %.2f %.2f]mm base  (%s)" %
                  (label, t[0], t[1], t[2], still_note))
        print("  " + S.msg + ("  -> %s" % MIC_CALIB_LOG))

    def log_mic_campaign_touch():
        """'9': log a HELD-OUT ground-truth point for the independent
        microscope-verification campaign (see mic_independent_verify.py and
        MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md) -- NOT for the extrinsic
        calibration fit. Writes to MIC_CAMPAIGN_LOG, a file
        mic_extrinsic_calib.py --fit never reads, so a campaign point can
        never accidentally contaminate the calibration it is meant to
        independently check (structural train/val separation, same
        principle as split_pairs_holdout.py).

        Workflow: physically drive the tip to rest on ANY sharp,
        unambiguous phantom feature (does not need to be an ink entry
        mark -- any clickable point spread across the field of view is
        fine, and ideally NOT one of the 8 calibration points), press '9',
        then take/transfer a clean microscope still and click that same
        point with mic_independent_verify.py --click. Label is
        auto-generated as 'campaign_<timestamp>' so repeated presses never
        collide.
        """
        if cur_tip() is None:
            S.msg = "no FrameEE yet -- cannot log campaign touch"; return
        label = "campaign_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        rec = {
            "label": label,
            "cur_tip_base_mm": cur_tip().tolist(),
            "ee_p_base_mm": _ee["p"].tolist(),
            "ee_q": _ee["q"].tolist(),
            "timestamp": time.time(),
            "timestamp_iso": datetime.datetime.now().isoformat(),
        }
        with open(MIC_CAMPAIGN_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        t = rec["cur_tip_base_mm"]
        S.msg = ("MIC CAMPAIGN TOUCH '%s' logged (HELD-OUT, not used for calib fit): "
                  "tip=[%.2f %.2f %.2f]mm base -- now take+transfer a microscope still and "
                  "click this point with mic_independent_verify.py --click" % (label, t[0], t[1], t[2]))
        print("  " + S.msg + ("  -> %s" % MIC_CAMPAIGN_LOG))

    def log_ring_offset_sample():
        """'0': append ONE paired (D405 ring-lock, microscope entry-mark)
        sample to RING_OFFSET_SAMPLES_LOG, for mic_ring_offset_verify.py
        --fit to later learn+validate (with leave-one-out CV, not a
        single anecdotal snapshot -- see that script's docstring) the
        fixed offset the 'x' key adds on top of the live ring lock.

        Unlike '1'-'8' (which need the TOOL TIP physically touching a
        landmark -- mic_perception.py's own docstring says the instrument
        is an unreliable, out-of-focus blob to auto-detect), this key
        needs NO physical motion at all: it just reads (a) the CURRENT
        live D405 ring lock and (b) the CURRENT freshest microscope still,
        auto-detecting the ink entry marks the exact same way 'y'/'e'
        already trust. So it is safe/cheap to press repeatedly, ideally
        across >=2 sessions (re-approach, re-lock 'z', press '0' again)
        to build up a real, non-self-referential validation set instead
        of trusting one lucky snapshot.

        Requires:
          - the D405 ring currently LOCKED (_ring_live["locked"]; press
            'z' first if not -- this key does not lock it for you)
          - MIC_EXTRINSIC_CALIB_NPZ already fit (mic_extrinsic_calib.py
            --fit --loo) -- without it there is no trusted base-frame
            position to compare the ring lock against, so this refuses
          - a fresh (<=MIC_LIVE_MAX_AGE_S old), clean microscope still in
            MIC_LIVE_STILLS_DIR, same staleness gate 'y' already applies
        """
        stale = calib_freshness_problems(phase="collect")
        if stale:
            S.msg = "0: REFUSED (stale calib): " + "; ".join(stale)
            print("  " + S.msg)
            return False
        if not _ring_live["locked"]:
            S.msg = "ring not locked -- press 'z' first (need a live D405 ring lock to pair)"
            print("  0: " + S.msg)
            return
        if not os.path.isfile(MIC_EXTRINSIC_CALIB_NPZ):
            S.msg = ("no %s -- run mic_extrinsic_calib.py --fit --loo first (need a trusted "
                      "microscope->base map before a ring-lock sample means anything)" %
                      MIC_EXTRINSIC_CALIB_NPZ)
            print("  0: " + S.msg)
            return
        preview = get_mic_preview()
        if preview is None:
            S.msg = "no stills in %s -- capture+transfer one first" % MIC_LIVE_STILLS_DIR
            print("  0: " + S.msg)
            return
        _thumb, rec, latest, mtime = preview
        age_s = time.time() - mtime
        if age_s > MIC_LIVE_MAX_AGE_S:
            S.msg = ("newest still is %.0fs old (>%.0fs limit) -- capture a FRESH one first" %
                      (age_s, MIC_LIVE_MAX_AGE_S))
            print("  0: " + S.msg)
            return
        if not rec["ok"]:
            S.msg = "mic_perception: plug/limbus not found in %s -- retake" % os.path.basename(latest)
            print("  0: " + S.msg)
            return

        sys.path.insert(0, MIC_AUTONOMY_DIR)
        import mic_extrinsic_calib as _mec
        mark_bases = {}
        for i in range(2):
            cx, cy = rec.get("mark%d_cx" % i), rec.get("mark%d_cy" % i)
            if cx is None:
                continue
            X, Y, Z = _mec.pixel_to_base_xy(cx, cy, MIC_EXTRINSIC_CALIB_NPZ)
            mark_bases[i] = [X, Y, Z]

        if not mark_bases:
            S.msg = "mic_perception found no ink marks in %s -- retake" % os.path.basename(latest)
            print("  0: " + S.msg)
            return

        out = {
            "mark0_theta_deg": float(np.degrees(np.arctan2(rec["mark0_cy"]-rec["plug_cy"], rec["mark0_cx"]-rec["plug_cx"]))) if rec.get("mark0_cx") is not None else None,
            "mark0_radius_px": float(np.hypot(rec["mark0_cx"]-rec["plug_cx"], rec["mark0_cy"]-rec["plug_cy"])) if rec.get("mark0_cx") is not None else None,
            "timestamp": time.time(),
            "timestamp_iso": datetime.datetime.now().isoformat(),
            "ring_base_mm": _ring_live["smoothed"].tolist(),
            "ring_jitter_mm": float(_ring_live["jitter"]),
            "ring_depth_src": _ring_live.get("depth_src"),
            "ring_n": int(_ring_live["n"]),
            "mic_still_path": latest,
            "mic_still_age_s": age_s,
            "occluded": bool(rec["occluded"]),
            "calib_source_npz": MIC_EXTRINSIC_CALIB_NPZ,
            # 2026-10-05: log the CURRENT live phantom normal (updated by 'n',
            # see PHANTOM_NORMAL) alongside this sample -- required by
            # mic_ring_offset_verify.py --fit-circle to build this sample's
            # tangent frame (tangent_frame_from_normal()). None if 'n' has
            # never been pressed this session; such samples still work for
            # the original --fit (fixed-vector) path, just not --fit-circle.
            "phantom_normal_base": (PHANTOM_NORMAL["normal"].tolist()
                                     if PHANTOM_NORMAL is not None else None),
            "phantom_normal_point_base": (PHANTOM_NORMAL["point"].tolist()
                                           if PHANTOM_NORMAL is not None else None),
        }
        for i in (0, 1):
            out["mark%d_base_mm" % i] = mark_bases.get(i)

        try:
            dbg_dir = data_path("ring_depth_debug")
            os.makedirs(dbg_dir, exist_ok=True)
            dbg_path = os.path.join(dbg_dir, "ring_%d.npz" % int(out["timestamp"] * 1000))
            np.savez_compressed(dbg_path, color=_ring_live.get("last_color"),
                                depth_mm=np.asarray(_ring_live.get("last_depth_mm"), np.float32),
                                cx=_ring_live["cx"], cy=_ring_live["cy"], r_px=_ring_live["r_px"],
                                fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
            out["ring_debug_npz"] = dbg_path
        except Exception as e:
            print("  0: ring debug dump failed (non-fatal): %s" % e)

        with open(RING_OFFSET_SAMPLES_LOG, "a") as fh:
            fh.write(json.dumps(out) + "\n")

        parts = ["mark%d=[%.1f %.1f %.1f]" % (i, *mark_bases[i]) for i in sorted(mark_bases)]
        S.msg = ("RING-OFFSET SAMPLE logged: ring=[%.1f %.1f %.1f]mm  %s%s%s" %
                  (out["ring_base_mm"][0], out["ring_base_mm"][1], out["ring_base_mm"][2],
                   "  ".join(parts), "  [OCCLUDED]" if rec["occluded"] else "",
                   "  [NO PHANTOM NORMAL -- press 'n' first if you want this sample usable by "
                   "--fit-circle]" if PHANTOM_NORMAL is None else ""))
        print("  " + S.msg + ("  -> %s" % RING_OFFSET_SAMPLES_LOG))

    def record_handeye_sample():
        """SHIFT+H: burst-average N frames' worth of (ArUco-on-shaft
        camera-frame position + depth, FrameEE pose) and APPEND one row to
        HANDEYE_PAIRS_CSV -- functionally identical to what the standalone
        handeye_capture_one.py does, except IN-PROCESS here so jogging
        (wasd/rf/kmjl/uo) and hand-eye sample recording can interleave in
        THIS SAME running instance. No more quit-jog-quit-capture-relaunch
        toggle dance with a second script (that dance exists only because
        handeye_calibrate.py's OWN interactive loop holds the camera AND has
        no jog keys of its own -- this script already holds the camera and
        already has jog keys, so it can just do both).

        Writes the exact CSV schema handeye_calibrate.py expects
        (cam_x,cam_y,cam_z,base_x,base_y,base_z,qx,qy,qz,qw), so once you
        have >=12 well-spread samples, just launch handeye_calibrate.py --
        it auto-resumes this CSV ("Resumed N pairs") -- and press 's' to
        solve. No jogging needed at that final step.

        STOP moving before pressing this -- like handeye_capture_one.py, it
        does not itself check the robot is still (that already surfaced a
        1.8mm-fit/23.9mm-LOO problem once, see handeye_calibrate.py's own
        docstring) -- it only warns AFTER the fact via jitter numbers, it
        does not block a jittery save. Read the printed jitter before
        trusting the sample; re-record (no need to delete anything first --
        just press SHIFT+H again from the same stationary pose) if it looks
        bad."""
        if _detect_aruco is None:
            S.msg = "cannot import detect_aruco from handeye_calibrate.py: %s" % _ARUCO_IMPORT_ERR
            return False
        if _ee["p"] is None or _ee["q"] is None:
            S.msg = "no FrameEE yet -- cannot record hand-eye sample"
            return False
        centers, p_samples, q_samples = [], [], []
        for _ in range(HANDEYE_SAMPLE_N_TARGET + 8):
            try:
                frames = pipe.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue
            frames = align.process(frames)
            color_f = frames.get_color_frame()
            depth_f = frames.get_depth_frame()
            if not color_f or not depth_f:
                continue
            color = np.asanyarray(color_f.get_data())
            center, _corners, _label = _detect_aruco(color)
            if center is None:
                continue
            u, v = int(round(center[0])), int(round(center[1]))
            vals = []
            for du in range(-2, 3):
                for dv in range(-2, 3):
                    d = depth_f.get_distance(u + du, v + dv) * 1000.0
                    if d > 0:
                        vals.append(d)
            if not vals:
                continue
            z = float(np.median(vals))
            x = (center[0] - intr.ppx) / intr.fx * z
            y = (center[1] - intr.ppy) / intr.fy * z
            centers.append((x, y, z))
            p_samples.append(_ee["p"].copy())
            q_samples.append(_ee["q"].copy())
            if len(centers) >= HANDEYE_SAMPLE_N_TARGET:
                break
        if len(centers) < 6:
            S.msg = ("hand-eye sample FAILED: only %d/%d valid ArUco+depth frames -- marker not "
                      "reliably visible/stable at this pose, reorient shaft toward camera and retry" %
                      (len(centers), HANDEYE_SAMPLE_N_TARGET))
            print("  " + S.msg)
            return False
        centers = np.array(centers); p_arr = np.array(p_samples); q_arr = np.array(q_samples)
        cam_pt = centers.mean(axis=0); cam_jitter = centers.std(axis=0)
        p_mean = p_arr.mean(axis=0); p_jitter = p_arr.std(axis=0)
        q_mean = q_arr.mean(axis=0); q_mean /= np.linalg.norm(q_mean)
        jitter_bad = np.any(p_jitter > 0.5) or np.any(cam_jitter > 3.0)

        os.makedirs(HANDEYE_DATA_DIR, exist_ok=True)
        new_file = not os.path.isfile(HANDEYE_PAIRS_CSV)
        with open(HANDEYE_PAIRS_CSV, "a") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow(["cam_x", "cam_y", "cam_z", "base_x", "base_y", "base_z", "qx", "qy", "qz", "qw"])
            w.writerow([cam_pt[0], cam_pt[1], cam_pt[2], p_mean[0], p_mean[1], p_mean[2],
                        q_mean[0], q_mean[1], q_mean[2], q_mean[3]])
        n_total = sum(1 for _ in open(HANDEYE_PAIRS_CSV)) - 1
        S.msg = ("HAND-EYE sample #%d recorded (n_valid=%d/%d, cam_jitter=%.2fmm, "
                  "p_jitter=%.2fmm)%s -> %s" % (
                      n_total, len(centers), HANDEYE_SAMPLE_N_TARGET,
                      float(cam_jitter.max()), float(p_jitter.max()),
                      "  !! WARNING: robot/marker doesn't look fully still, consider re-recording"
                      if jitter_bad else "", HANDEYE_PAIRS_CSV))
        print("  " + S.msg)
        return True

    # 2026-10-06: found via `sudo py-spy dump` (6 live samples of the running
    # process while the operator reported the tip dot still not tracking in
    # real time even after the cv2.setNumThreads(1) and ROS queue_size=1
    # fixes above): 4/6 samples caught the MainThread INSIDE either
    # update_live_ring_detection() -> detect_ring() -> _segment_ring_from_
    # mask() -> _ring_try_candidates() -> kasa_circle_fit() (a numpy circle
    # fit over multiple candidate subsets), or mic_perception.locate() ->
    # find_plug_circle()/detect_entry_marks() (HoughCircles + a radial
    # profile scan). Both are real, substantial CPU costs -- NOT bugs by
    # themselves, just classical-CV work that was never designed to run on
    # literally every single D405 frame (up to 30Hz) on a machine that is
    # ALSO running gscam/mic_live_capture_ros.py/eye_robot_example at the
    # same time. mic_perception.locate() already only reruns on a new file
    # (see get_mic_preview()'s path/mtime check), but
    # update_live_ring_detection() had NO throttle at all -- it ran
    # unconditionally on every grab_show() call. Capping it to
    # RING_DETECT_MIN_INTERVAL_S between actual re-detections (holding the
    # last result, exactly like the mic-still cache does) trades a small
    # amount of ring-lock smoothing latency for a much more responsive
    # main loop -- the tip overlay/cur_tip() read and the imshow/waitKey
    # below are completely unaffected by this and still run every frame.
    RING_DETECT_MIN_INTERVAL_S = float(os.environ.get("RING_DETECT_MIN_INTERVAL_S", "0.1"))  # 10Hz cap
    _ring_detect_last_t = [0.0]
    # 2026-10-06: same py-spy session also caught 11.8% of total MainThread
    # time inside detect_aruco() (fast=True), which -- like the ring
    # detector above -- runs completely unconditionally on every single
    # grab_show() call purely to populate a passive "is a marker visible"
    # status line (SHIFT+H's own accurate capture is a separate, deliberate
    # call elsewhere and is NOT throttled by this). 10Hz is still far more
    # than a human needs to see that status update live.
    ARUCO_DETECT_MIN_INTERVAL_S = float(os.environ.get("ARUCO_DETECT_MIN_INTERVAL_S", "0.1"))  # 10Hz cap
    _aruco_detect_last_t = [0.0]
    # 2026-10-06 (pt.3): the user keeps reporting "tip dot can't track in
    # real time" even after 6 verified, measured fixes (cv2 thread
    # contention, ROS queue_size, 3x throttles, async mic locate) --
    # latest report floats "maybe it's the lighting". Important: the green
    # TIP dot (COLOR_TIP, drawn via cur_tip()) is pure robot-kinematics +
    # a static offset off /FrameEE @ 193Hz -- it NEVER touches
    # mic_perception.py, lighting, or the microscope at all, so lighting
    # cannot be the cause of THIS dot lagging specifically. The only two
    # things that COULD still make it visually lag are (a) this main
    # grab_show() loop's own achieved FPS being low (redraw rate), or (b)
    # /FrameEE itself arriving stale despite queue_size=1. Rather than
    # guess again, put a live, on-screen, objective loop-FPS readout
    # directly in front of the user so "is it actually slow right now"
    # stops being something only py-spy after the fact can answer.
    _fps_tracker = {"t_win": time.time(), "n_win": 0, "fps": 0.0}

    def _tick_fps():
        _fps_tracker["n_win"] += 1
        now = time.time()
        dt = now - _fps_tracker["t_win"]
        if dt >= 0.5:
            _fps_tracker["fps"] = _fps_tracker["n_win"] / dt
            _fps_tracker["t_win"] = now
            _fps_tracker["n_win"] = 0
        return _fps_tracker["fps"]

    _calib_fresh_cache = {"t": 0.0, "probs": []}

    def grab_show():
        frames = pipe.wait_for_frames(timeout_ms=2000)
        frames = align.process(frames)  # depth pixel coords -> color pixel coords
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        img = np.asanyarray(color_f.get_data()).copy() if color_f else np.zeros((480, 640, 3), np.uint8)
        if color_f and depth_f:
            _now_rd = time.time()
            if (_now_rd - _ring_detect_last_t[0]) >= RING_DETECT_MIN_INTERVAL_S:
                _ring_detect_last_t[0] = _now_rd
                update_live_ring_detection(img, depth_f)
        _now_ad = time.time()
        if color_f and _detect_aruco is not None and \
                (_now_ad - _aruco_detect_last_t[0]) >= ARUCO_DETECT_MIN_INTERVAL_S:
            _aruco_detect_last_t[0] = _now_ad
            try:
                # 2026-09-28: fast=True -- this per-frame live status call was
                # measured (via py-spy on a genuinely "frozen" session) to be
                # THE keyboard-teleop-freeze root cause: APRILTAG corner
                # refinement is ~9-10x slower than SUBPIX here, and running
                # every grab_show() frame stretched the loop to ~0.4-2s/frame,
                # starving cv2.waitKey(20). See detect_aruco()'s fast= param
                # docstring in handeye_calibrate.py. record_handeye_sample()'s
                # deliberate SHIFT+H capture below keeps the accurate default.
                # 2026-10-06: ALSO rate-limited now (see
                # ARUCO_DETECT_MIN_INTERVAL_S above) -- fast=True alone
                # wasn't enough; a later py-spy session on THIS machine still
                # attributed 11.8% of total MainThread time to this call.
                a_center, a_corners, a_label = _detect_aruco(img, fast=True)
            except Exception as _e:
                a_center, a_corners, a_label = None, None, None
                _aruco_live["status_msg"] = "ArUco detect error (non-fatal): %s" % _e
            _aruco_live["found"] = a_center is not None
            _aruco_live["center"] = a_center
            _aruco_live["corners"] = a_corners
            _aruco_live["label"] = a_label
            if a_center is not None:
                _aruco_live["status_msg"] = "%s visible at (%.0f,%.0f)" % (a_label, a_center[0], a_center[1])
            elif "error" not in _aruco_live["status_msg"]:
                _aruco_live["status_msg"] = "marker not visible -- reposition/reorient shaft toward camera"
        _pres = _present["on"]
        MIC_THUMB_W["w"] = 472 if _pres else 260
        view = draw_present(img) if _pres else draw(img)
        fps_now = _tick_fps()
        fps_col = HUD_DIM if fps_now >= 15 else (HUD_WARN if fps_now >= 8 else HUD_BAD)
        # 2026-10-06 (pt.3): see _cb_ee's "t" stamp comment -- this tells
        # the user/operator, live, whether the green TIP dot's underlying
        # /FrameEE data is fresh (this number should normally stay well
        # under ~50ms given FrameEE publishes at ~193Hz) or whether the ROS
        # callback itself is being starved despite queue_size=1.
        ee_age_s = (time.time() - _ee["t"]) if _ee.get("t") is not None else None
        ee_ms = ee_age_s * 1000.0 if ee_age_s is not None else None
        ee_col = HUD_BAD if (ee_ms is None or ee_ms > 300) else (HUD_WARN if ee_ms > 100 else HUD_DIM)
        corr_mag = float(np.linalg.norm(ORIGIN_CORRECTION))
        tip_z_mm = None
        if handeye_ok and cur_tip() is not None:
            cam_tip = R_he.T.dot(cur_tip() - t_he)
            if cam_tip[2] > 1e-3:
                tip_z_mm = float(cam_tip[2])
        z_bad = tip_z_mm is not None and not (ORIGIN_CLICK_Z_MIN_M * 1000.0 <= tip_z_mm <=
                                              ORIGIN_CLICK_Z_MAX_M * 1000.0)
        corr_col = HUD_BAD if (corr_mag > 20.0 or z_bad) else HUD_DIM
        if not _pres:
            perf = [("%.0f fps" % fps_now, fps_col),
                    (("EE %.0fms" % ee_ms) if ee_ms is not None else "EE --", ee_col),
                    ("corr %.1f" % corr_mag, corr_col),
                    (("Z %.0f" % tip_z_mm) if tip_z_mm is not None else "Z --", HUD_BAD if z_bad else HUD_DIM)]
            H_, W_ = view.shape[:2]
            px = W_ - 10
            for txt, col in reversed(perf):
                px -= hud_tw(txt, 0.36)
                hud_text(view, txt, (px, H_ - 10), 0.36, col)
                px -= 12
        _now_cf = time.time()
        if _now_cf - _calib_fresh_cache["t"] > 2.0:
            _calib_fresh_cache["t"] = _now_cf
            _calib_fresh_cache["probs"] = calib_freshness_problems()
        if _calib_fresh_cache["probs"] and not _pres:
            short = [p.split(" ")[0] for p in _calib_fresh_cache["probs"]]
            ctxt = "CALIB needed: " + ", ".join(short)
            cw = min(hud_tw(ctxt, 0.38) + 20, W_ - 290)
            hud_panel(view, 4, 44, 4 + cw, 62, 0.6)
            cv2.circle(view, (12, 53), 4, HUD_BAD, -1, cv2.LINE_AA)
            hud_text(view, ctxt, (20, 57), 0.38, (150, 150, 255), max_w=cw - 20)
        if S.rec:
            _now_rec = time.time()
            if _rec_clock["t0"] is None or _now_rec - _rec_clock["last"] > REC_RESUME_GAP_S:
                _rec_clock["t0"] = _now_rec - _rec_clock["n"] / REC_FPS
            _rec_clock["last"] = _now_rec
            _due = int((_now_rec - _rec_clock["t0"]) * REC_FPS) + 1
            if _due > _rec_clock["n"]:
                _frame = view if view.shape[1] == 1280 else cv2.resize(view, (1280, 720), interpolation=cv2.INTER_CUBIC)
                for _ in range(_due - _rec_clock["n"]):
                    vw.write(_frame)
                _rec_clock["n"] = _due
        cv2.imshow(win, view)
        return cv2.waitKey(20) & 0xFF

    def auto_move(goal, why):
        """CHANGED (2026-08-23, pt.17): added the same two anti-stall
        protections auto_orient_about_rcm() already had (see pt.1) but this
        function never got: a stall detector and a minimum-speed floor.
        Reported symptom: RETRACT (the one auto_move() call that travels
        farthest, HOME<->deep-in-tissue) sometimes never reached HOME and
        never printed any error -- it just hung with S.msg frozen on
        "RETRACT ...". Root cause: this while-loop's ONLY exit conditions
        were "goal reached" or "space/q pressed" -- if the robot physically
        cannot reduce dnorm any further for any reason (joint/cartesian soft
        limit, cable drag, static friction/backlash exceeding the torque the
        tapered-to-near-zero velocity command implies right at the tail of a
        long move -- the same classic "P-only, no floor, near a dead zone"
        failure mode already diagnosed and fixed for rotation in pt.1, just
        never ported to translation), it will publish velocity commands
        forever with zero feedback that anything is wrong. MAX_TRAVEL_MM
        (pt.7) is disabled by default now too, so that old runaway-abort
        safety net does not catch this either. Fix mirrors pt.1 exactly:
        MOVE_STALL_ITERS consecutive ticks with dnorm not decreasing ->
        zero() + abort with an honest "STALLED" message instead of hanging;
        MOVE_MIN_SPEED floors the linear-taper speed so it can never decay
        to something real friction/backlash can beat.
        """
        if _ee["p"] is None:
            S.msg = "no FrameEE, abort " + why; return False
        start = _ee["p"].copy()
        t_start = time.time()
        S.msg = "%s ..." % why
        print("  %s -> [%.2f %.2f %.2f] (tip-space)" % (why, goal[0], goal[1], goal[2]))
        rate = rospy.Rate(100)
        prev_dnorm = None
        stall_count = 0
        while not rospy.is_shutdown():
            cur_ee = _ee["p"]
            if cur_ee is None:
                zero(); S.msg = "lost FrameEE, abort"; return False
            if MAX_TRAVEL_MM > 0 and np.linalg.norm(cur_ee - start) > MAX_TRAVEL_MM:
                zero(); S.msg = "MAX_TRAVEL exceeded, abort " + why; print("  " + S.msg)
                log_precision(why, result="max_travel_abort", err_mm="", elapsed_s=round(time.time() - t_start, 2))
                return False
            cur = cur_ee + tip_offset_vec()
            err = goal - cur; dnorm = np.linalg.norm(err)
            if dnorm < POS_TOL_MM:
                zero(); S.msg = "%s reached (%.2fmm)" % (why, dnorm); print("  " + S.msg)
                log_precision(why, result="reached", err_mm=round(float(dnorm), 3),
                               elapsed_s=round(time.time() - t_start, 2))
                return True
            # Progress is judged against the best distance so far: with noisy FrameEE a
            # per-iteration comparison keeps resetting, so a robot stopped at a limit
            # never registered as stalled (trial 5, 2026-10-12: RETRACT hung for 90 s).
            if prev_dnorm is None or dnorm < best_dnorm - 0.05:
                best_dnorm = dnorm
                stall_count = 0
            else:
                stall_count += 1
            if stall_count > MOVE_STALL_ITERS:
                zero()
                S.msg = ("%s STALLED at %.2fmm (error not decreasing for %.1fs) -- "
                         "likely hit a joint/cartesian limit or mechanical "
                         "obstruction; aborting instead of hanging forever" %
                         (why, dnorm, MOVE_STALL_ITERS / 100.0))
                print("  " + S.msg)
                log_precision(why, result="stalled", err_mm=round(float(dnorm), 3),
                               elapsed_s=round(time.time() - t_start, 2))
                return False
            prev_dnorm = dnorm
            u = err / dnorm
            if why in PIERCE_MOVES:
                mag = min(PIERCE_SPEED, PIERCE_SPEED * (dnorm / PIERCE_TAPER_MM))
            elif why == "RETRACT_AXIAL":
                mag = min(RETRACT_AXIAL_SPEED, RETRACT_AXIAL_SPEED * (dnorm / PIERCE_TAPER_MM))
            else:
                mag = min(SPEED, SPEED * (dnorm / 3.0))
            mag = max(mag, MOVE_MIN_SPEED)
            pub.publish(Vector3(u[0] * mag, u[1] * mag, u[2] * mag))
            k = grab_show()
            if k == ord(' '):
                zero(); S.msg = "STOP (space)"; print("  " + S.msg)
                log_precision(why, result="stopped", err_mm=round(float(dnorm), 3),
                               elapsed_s=round(time.time() - t_start, 2))
                return False
            if k in (ord('q'), 27):
                zero(); raise KeyboardInterrupt
            rate.sleep()
        zero()
        return False

    def auto_orient_about_rcm(target_dir, why, pivot_point=None, compensate=True):
        """Rotate the tool until NEEDLE_AXIS_LOCAL aligns with target_dir,
        pivoting about `pivot_point` if given, else the latched (surface)
        RCM point. `pivot_point` has three call sites, each pivoting about
        a different fixed point:
          - PLUNGE re-orient: the NEW internal RCM (tip position at the end
            of the oblique TUNNEL), not the original surface RCM -- using
            the wrong pivot here would tear the scleral tunnel sideways
            instead of re-orienting in place at its base.
          - 't'/ORIENT (2026-08-18 reorder, see file docstring pt.10): the
            tip's OWN current position, captured once right before this
            call -- the compensation (see pt.11 for the EE-origin-vs-tip
            lever fix) keeps the tip circling in place around where it
            started, a true "rotate without displacing" at the hover point,
            instead of sweeping a wide arc about the far-away surface RCM
            the way orienting from HOME would.
          - no pivot_point given: falls back to the latched surface RCM
            (legacy behavior, used only if some caller still needs it).

        NOTE (2026-08-18, pt.11): the lever used below is EE_origin - pivot,
        NOT tip - pivot -- see file docstring pt.11 for why tip-pivot was a
        real bug (it silently under/over-compensated by the tip offset,
        most visibly for the 't' case above where pivot ~= tip so the old
        lever was ~0 throughout, i.e. no compensation at all).

        CONTROL LAW CHANGE vs baseline (see file-level docstring for the
        2026-08-13 log evidence motivating this):
          OLD: omega = omega_dir * ASPEED * min(1, err_rad/5deg)   (pure P,
               band=5deg, NO floor) -- can stall arbitrarily close to zero
               commanded speed as err_rad -> small, if static friction/
               backlash exceeds the torque implied by that tiny speed
               command. This is exactly a classic "P-only near a dead zone"
               failure mode.
          NEW: omega = omega_dir * clip(P_term + I_term, floor, ASPEED)
               - P_term = ASPEED * min(1, err_rad/ORIENT_EASE_DEG)  (same
                 shape as before, so far-field behavior/speed is unchanged)
               - I_term accumulates err_rad*dt only INSIDE the ease band
                 (avoids classical integral windup during the fast coarse
                 phase), clamped to +/-ORIENT_KI_MAX_FRAC*ASPEED
               - floor = ORIENT_MIN_OMEGA whenever err_deg > ORIENT_TOL_DEG,
                 so the commanded speed can never decay to something the
                 real actuator can't overcome -- this directly targets the
                 "stalls a few degrees short" failure mode.
          Tolerance ORIENT_TOL_DEG default tightened 1.0 -> 0.3deg (only
          reasonable now that stalling near the target is fixed).

        `compensate=False` (2026-08-18, pt.14; REVERTED 2026-08-23, pt.17 --
        see pt.17 changelog entry for the quantified reasoning): used to skip
        the omega x lever linear compensation entirely for 't', on the
        theory that real hardware cannot hold the tip exactly on the pivot
        anyway so a compensation that cannot reach its own goal is just
        extra unpredictable motion for no benefit. That theory undersold how
        LARGE the resulting displacement actually is: with compensate=False,
        v_lin is commanded as exactly zero, so the EE origin does not move
        at all while the tool rotates about IT, not about the tip -- the tip
        (offset from the EE origin by TIP_D, ~35.8mm on this rig's current
        pivot calibration) sweeps a chord of 2*|TIP_D|*sin(angle/2), e.g.
        ~18.5mm for a 30deg ORIENT. That is not a small residual, it is a
        systematic, fully-predictable-in-advance displacement of the exact
        scale pt.14 was trying to avoid. pt.11's lever-formula fix (tip-pivot
        -> EE_origin-pivot) makes compensate=True's v_lin = omega x lever
        feedforward the mathematically EXACT continuous-time solution for
        keeping the tip on `pivot_point` while the tool rotates (not merely
        "first order per tick" as pt.13/pt.14 characterized it -- that
        caveat is about DISCRETIZATION at 100Hz plus real-world actuator lag/
        encoder noise/calibration bias, which are all much smaller than the
        ~18.5mm compensate=False sweep). So 't' now calls with
        compensate=True again (the default), and gets the ADDITIONAL
        closed-loop trim below on top of the feedforward, to correct
        whatever smaller residual drift those real-world effects still
        cause.

        NEW (2026-08-23, pt.17): closed-loop tip-position-HOLD trim, on top
        of (not instead of) the omega x lever feedforward, active whenever
        compensate=True. `hold_point` is captured ONCE, from the ACTUAL
        current tip position at the moment this call starts (not from
        `pivot_point`/`rcm_pt`, though for every current call site the two
        are within POS_TOL_MM of each other at t=0 -- 't' passes
        pivot_point=cur_tip() itself, and PLUNGE's reorient runs right after
        TUNNEL parked the tip AT S.internal_rcm). Every tick, the trim adds
        v_fb = clip(ORIENT_HOLD_KP * (hold_point - cur_tip()), <=
        ORIENT_HOLD_MAX_MM_S) to the feedforward v_lin. This is standard
        feedforward+feedback: the feedforward (omega x lever) handles the
        bulk of the required motion from first principles, while this small,
        capped proportional trim mops up whatever the feedforward's
        real-world imperfections (actuator lag, encoder noise, residual
        hand-eye/pivot-calibration bias -- pt.13's list) leave behind,
        directly from the OBSERVED tip drift rather than trying to model
        each error source. The cap (ORIENT_HOLD_MAX_MM_S, default 2.0mm/s)
        keeps this a trim, not a second controller fighting the first: if
        `hold_point` ever turns out to be a bad reference (e.g. pivot_point
        was itself far from the true tip at t=0), the trim saturates at a
        small speed rather than producing a large, confusing correction
        motion.

        Everything else (inertial-frame angular velocity convention,
        stall-abort safety net, timeout, space-to-stop) is UNCHANGED.

        FIXED (2026-09-28, pt.18, real bug -- reported symptom: "same RCM
        point, first-time oblique-insertion orientation sometimes ends up
        completely different between two consecutive attempts, and/or it
        picks CW when 5deg of CCW would have sufficed, spins the long way,
        and gets stuck (visibly hits the STALLED abort below)"):

        Root cause traced all the way into the C++ side (SHER_I2RIS_teleoperate
        robotModel::ComputeJacobianTip(), ROBOT_MODEL "2.1" branch, confirmed
        that IS what this rig's robotVersion.h selects): this robot has 5
        actuated joints, and only TWO of them (q3, q4) contribute any angular
        velocity at all. Reading the Jacobian's own coefficients directly (not
        guessing): qdot3's angular contribution is the FIXED base-frame axis
        (0,1,0); qdot4's is (cos(q3), 0, -sin(q3)) -- a direction in the
        base-frame X-Z plane that itself rotates as q3 changes. Together
        these span only a 2-D "instantaneously achievable rotation axis"
        plane, NOT the full 3-D rotation space.

        The old code below computed `rotvec = rotation_between(v_cur,
        target_dir)` -- the mathematically-shortest full-3D rotation -- and
        published it directly as omega, silently assuming all 3 rotational
        DOF were free. Whenever that ideal 3-D axis had a large component
        OUTSIDE the two-jointed plane above (which direction it's "large" or
        "small" in depends on the CURRENT q3, i.e. on where the robot
        happened to be left after whatever ran before this ORIENT call --
        exactly why the same nominal RCM point could converge to a
        different final orientation on different attempts, and why it was
        "sometimes fine, sometimes stuck spinning the wrong way": the
        un-achievable component doesn't just get ignored for free, fighting
        it can visibly manifest as the wrong net rotation direction on the
        two axes that ARE real, and the STALLED detector's own comment
        ("angular vel frame convention may be wrong") was actually pointing
        at this).

        FIX: every tick, project `rotvec` onto the CURRENT achievable plane
        (using live q3 from /SHER21/eye_robot/JointPosition, see
        JOINTPOS_TOPIC/_cb_jointpos/_robot_q3) before turning it into a unit
        direction -- i.e. drive only the component of the desired rotation
        the hardware can actually realize AT THIS INSTANT, instead of a
        3-D vector part of which is unreachable and was apparently
        corrupting the achievable part too. `err_deg`/stall/timeout still
        use the TRUE (unprojected) 3-D angle -- the goal is still full 3-D
        convergence, which two sequentially-driven joints CAN reach over
        time (q3 itself changing re-orients the q4 axis), just not via one
        instantaneous straight-line 3-D rotation. If the live q3 feed is
        stale/absent, falls back to the old unprojected behavior (printed
        once, not silently) rather than guessing.
        """
        if _ee["q"] is None or _ee["p"] is None:
            S.msg = "no FrameEE, abort " + why; return False
        if pivot_point is not None:
            rcm_pt = np.asarray(pivot_point, float).copy()
        else:
            if not rcm_ok():
                S.msg = "RCM not latched, abort " + why; return False
            rcm_pt = cur_rcm().copy()
        hold_point = cur_tip().copy()  # pt.17: actual tip pos at t=0, held via closed-loop trim below
        S.msg = "%s ..." % why
        print("  %s: target_dir(base)=[%.3f %.3f %.3f]  rcm=[%.2f %.2f %.2f]" %
              (why, target_dir[0], target_dir[1], target_dir[2], rcm_pt[0], rcm_pt[1], rcm_pt[2]))
        rate = rospy.Rate(100)
        dt = 1.0 / 100.0
        prev_err_deg = None
        stall_count = 0
        start_t = time.time()
        integral = 0.0  # rad*s accumulated error, only inside the ease band
        ease_rad = np.radians(ORIENT_EASE_DEG)
        ki_max = ORIENT_KI_MAX_FRAC * ASPEED
        # pt.22 (2026-09-28, diagnosing a reported pt.18-fix regression --
        # "rotation direction still clearly wrong, spins a big loop before
        # coming back"): buffer a per-tick trace IN MEMORY (never touch disk
        # inside the 100Hz loop -- that would risk introducing real control-
        # loop timing jitter on a live robot) and flush it to ORIENT_TRACE_CSV
        # once, on whichever exit path this call takes. Lets a REAL captured
        # attempt be inspected tick-by-tick after the fact (is q3 actually
        # fresh throughout? does the chosen omega_dir ever flip sign in a
        # way rotvec's raw 3-D direction doesn't explain? etc.) instead of
        # reasoning about it from memory/theory alone.
        _trace = []

        def _flush_orient_trace(result):
            try:
                with open(ORIENT_TRACE_CSV, "w") as fh:
                    fh.write("# why=%s result=%s captured_at=%s\n" %
                             (why, result, datetime.datetime.now().isoformat()))
                    fh.write("t_s,err_deg,q3_deg,q3_fresh,comp_q3_deg,comp_q4_deg,"
                             "omega_dir_x,omega_dir_y,omega_dir_z\n")
                    for row in _trace:
                        fh.write(",".join("" if v is None else str(v) for v in row) + "\n")
            except Exception as e:
                print("  (orient trace write failed, non-fatal: %s)" % e)

        while not rospy.is_shutdown():
            R_ee = quat_to_R(_ee["q"])
            v_cur = R_ee.dot(NEEDLE_AXIS_LOCAL)
            v_cur = v_cur / np.linalg.norm(v_cur)
            rotvec = rotation_between(v_cur, target_dir)
            err_rad = np.linalg.norm(rotvec)
            err_deg = np.degrees(err_rad)
            if err_deg < ORIENT_TOL_DEG:
                zero(); S.msg = "%s reached (%.2fdeg)" % (why, err_deg); print("  " + S.msg)
                log_precision(why, result="reached", err_deg=round(float(err_deg), 3),
                               elapsed_s=round(time.time() - start_t, 2),
                               insert_angle_deg=INSERT_ANGLE_DEG,
                               azimuth_deg=("auto" if INSERT_AZIMUTH_DEG is None else INSERT_AZIMUTH_DEG))
                if VISION_ANGLE_CHECK:
                    try:
                        frames = pipe.wait_for_frames(timeout_ms=500)
                        frames = align.process(frames)  # depth pixel coords -> color pixel coords
                        color_f = frames.get_color_frame()
                        depth_f = frames.get_depth_frame()
                        if color_f and depth_f:
                            snap = np.asanyarray(color_f.get_data()).copy()
                            vmsg, vfields = vision_angle_sanity_check_3d(snap, depth_f, target_dir, rcm_pt)
                            if vmsg:
                                S.last_vis_check_msg = vmsg
                                print("  " + vmsg)
                            if vfields:
                                # snapshot _ee["q"] NOW (not whenever log_precision
                                # happens to run) -- see PRECISION_LOG_FIELDS comment;
                                # this is what lets vis3d_err_deg be traced back to a
                                # specific EE-frame axis hypothesis after the fact.
                                q_now = _ee["q"]
                                if q_now is not None:
                                    vfields["ee_qx"], vfields["ee_qy"], vfields["ee_qz"], vfields["ee_qw"] = \
                                        (round(float(v), 6) for v in q_now)
                                log_precision(why, result="reached_vischeck", **vfields)
                    except Exception as e:
                        print("  (vision-3D check skipped: %s)" % e)
                _flush_orient_trace("reached")
                return True
            if ORIENT_TIMEOUT_S > 0 and time.time() - start_t > ORIENT_TIMEOUT_S:
                zero(); S.msg = "%s TIMEOUT (%.1fdeg remaining)" % (why, err_deg)
                print("  " + S.msg)
                log_precision(why, result="timeout", err_deg=round(float(err_deg), 3),
                               elapsed_s=round(time.time() - start_t, 2),
                               insert_angle_deg=INSERT_ANGLE_DEG,
                               azimuth_deg=("auto" if INSERT_AZIMUTH_DEG is None else INSERT_AZIMUTH_DEG))
                _flush_orient_trace("timeout")
                return False
            if prev_err_deg is not None and err_deg > prev_err_deg - 1e-3:
                stall_count += 1
            else:
                stall_count = 0
            if stall_count > ORIENT_STALL_ITERS:
                zero()
                S.msg = ("%s STALLED at %.1fdeg (error not decreasing) -- angular vel frame "
                         "convention may be wrong, or hit a joint limit; aborting" % (why, err_deg))
                print("  " + S.msg)
                log_precision(why, result="stalled", err_deg=round(float(err_deg), 3),
                               elapsed_s=round(time.time() - start_t, 2),
                               insert_angle_deg=INSERT_ANGLE_DEG,
                               azimuth_deg=("auto" if INSERT_AZIMUTH_DEG is None else INSERT_AZIMUTH_DEG))
                _flush_orient_trace("stalled")
                return False
            prev_err_deg = err_deg

            # pt.18 (2026-09-28): project rotvec onto the CURRENT achievable
            # 2-D rotation-axis plane (q3/q4 only, see docstring pt.18) before
            # picking a direction -- see docstring for why the un-projected
            # 3-D rotvec was the real root cause of the reported wrong-
            # direction / gets-stuck / inconsistent-orientation bug.
            q3_now = _robot_q3["val"]
            q3_fresh = (q3_now is not None) and (time.time() - _robot_q3["t"] < ORIENT_Q3_MAX_AGE_S)
            if q3_fresh:
                axis_q3 = np.array([0.0, 1.0, 0.0])
                axis_q4 = np.array([np.cos(q3_now), 0.0, -np.sin(q3_now)])
                comp_q3 = np.dot(rotvec, axis_q3)
                comp_q4 = np.dot(rotvec, axis_q4)
                rotvec_feasible = comp_q3 * axis_q3 + comp_q4 * axis_q4
                feas_norm = np.linalg.norm(rotvec_feasible)
                if feas_norm > 1e-6:
                    omega_dir = rotvec_feasible / feas_norm
                else:
                    # rotvec is (near-)perpendicular to the plane the hardware
                    # can move in RIGHT NOW -- committing to the raw 3-D
                    # direction here would command almost nothing achievable
                    # (exactly the "stuck" failure mode this fix targets).
                    # Nudge along whichever real axis has the larger raw
                    # alignment with rotvec, which changes q3 and therefore
                    # re-orients axis_q4 for the next tick, rather than
                    # stalling in place.
                    omega_dir = axis_q3 if abs(comp_q3) >= abs(comp_q4) else axis_q4
                    if np.dot(omega_dir, rotvec) < 0:
                        omega_dir = -omega_dir
            else:
                comp_q3, comp_q4 = None, None  # pt.22 trace: keep column count fixed regardless of branch
                if not S.warned_no_q3:
                    print("  !! WARNING: no live q3 feed (%s) -- ORIENT using UNPROJECTED "
                          "3D rotation (pre-2026-09-28 pt.18 behavior; can pick a direction "
                          "the hardware cannot actually realize this instant)" % JOINTPOS_TOPIC)
                    S.warned_no_q3 = True
                omega_dir = rotvec / max(err_rad, 1e-9)
            _trace.append((round(time.time() - start_t, 3), round(err_deg, 3),
                           None if q3_now is None else round(float(np.degrees(q3_now)), 3),
                           bool(q3_fresh),
                           None if comp_q3 is None else round(float(np.degrees(comp_q3)), 3),
                           None if comp_q4 is None else round(float(np.degrees(comp_q4)), 3),
                           round(float(omega_dir[0]), 4), round(float(omega_dir[1]), 4),
                           round(float(omega_dir[2]), 4)))
            p_term = ASPEED * min(1.0, err_rad / max(ease_rad, 1e-9))
            if err_rad < ease_rad:
                integral += err_rad * dt
            else:
                integral = 0.0
            i_term = np.clip(ORIENT_KI * integral, -ki_max, ki_max)
            speed = p_term + i_term
            if err_deg > ORIENT_TOL_DEG:
                speed = max(speed, ORIENT_MIN_OMEGA)
            speed = min(speed, ASPEED_MAX)
            omega = omega_dir * speed

            if compensate:
                ee_now = _ee["p"]
                lever = ee_now - rcm_pt  # EE origin, NOT tip -- see docstring pt.11 (2026-08-18 fix)
                v_lin = np.cross(omega, lever)
                # pt.17 (2026-08-23): closed-loop trim on top of the feedforward
                # above, correcting residual drift of the tip away from
                # hold_point (its own actual position at t=0) using the
                # DIRECTLY OBSERVED error rather than trying to model its
                # source (actuator lag / encoder noise / calibration bias).
                tip_now = ee_now + tip_offset_vec()
                v_fb = ORIENT_HOLD_KP * (hold_point - tip_now)
                fb_speed = np.linalg.norm(v_fb)
                if fb_speed > ORIENT_HOLD_MAX_MM_S:
                    v_fb = v_fb * (ORIENT_HOLD_MAX_MM_S / fb_speed)
                v_lin = v_lin + v_fb
                speed_lin = np.linalg.norm(v_lin)
                if speed_lin > SPEED_MAX:
                    v_lin = v_lin * (SPEED_MAX / speed_lin)
            else:
                v_lin = np.zeros(3)  # pure rotation, no compensation attempt (legacy path, no longer used by 't')
            pub.publish(Vector3(*v_lin))
            pub_a.publish(Vector3(*omega))
            k = grab_show()
            if k == ord(' '):
                zero(); S.msg = "STOP (space)"; print("  " + S.msg)
                log_precision(why, result="stopped", err_deg=round(float(err_deg), 3),
                               elapsed_s=round(time.time() - start_t, 2),
                               insert_angle_deg=INSERT_ANGLE_DEG,
                               azimuth_deg=("auto" if INSERT_AZIMUTH_DEG is None else INSERT_AZIMUTH_DEG))
                _flush_orient_trace("stopped")
                return False
            if k in (ord('q'), 27):
                _flush_orient_trace("interrupted")
                zero(); raise KeyboardInterrupt
            rate.sleep()
        zero()
        _flush_orient_trace("ros_shutdown")
        return False

    print(__doc__)
    if pub.get_num_connections() == 0:
        print("  !! no subscriber on %s -- bringup + CM_OCT_CALIBCONTROL?" % VEL_TOPIC)
    if not rcm_ok():
        print("  !! RCM not set yet -- point the camera at the phantom's red ring and press 'z' "
              "(live in-process auto-detect, no separate terminal needed -- watch the 'ring[z]:' "
              "status line), or latch Mode 2/3 on the GUI for manual RCM before g/i")
    if not handeye_ok:
        print("  !! hand-eye npz missing (%s) -- overlay dots disabled, control still works" % NPZ)
    print("  precision log -> %s" % PRECISION_LOG)

    jog = {"lin": None, "ang": None, "until": 0.0}
    LIN = {ord('w'): (1, 0, 0), ord('s'): (-1, 0, 0), ord('a'): (0, 1, 0),
           ord('d'): (0, -1, 0), ord('r'): (0, 0, 1), ord('f'): (0, 0, -1)}
    ANG = {ord('k'): (1, 0, 0), ord('m'): (-1, 0, 0),
           ord('j'): (0, 1, 0), ord('l'): (0, -1, 0),
           ord('u'): (0, 0, 1), ord('o'): (0, 0, -1)}
    try:
        while not rospy.is_shutdown():
            k = grab_show()
            now = time.time()
            if k in (ord('g'), ord('t'), ord('i'), ord('p')):
                from runtime_session import motion_problems
                _problems = motion_problems()
                if _problems:
                    zero()
                    S.msg = "REFUSED: " + "; ".join(_problems)
                    print(S.msg)
                    continue
            if k == 255:
                if now < jog["until"] and (jog["lin"] is not None or jog["ang"] is not None):
                    lv = jog["lin"] or (0, 0, 0)
                    av = jog["ang"] or (0, 0, 0)
                    ang_cmd = [c * ASPEED for c in av]
                    lin_cmd = [c * SPEED for c in lv]
                    # 2026-08-18: manual ANGULAR jog (k/m/j/l/u/o) used to send
                    # av with ZERO linear velocity -- i.e. it rotated about
                    # whatever pivot the robot's OWN low-level controller
                    # defaults to, not about our tracked RCM. That is exactly
                    # the "rotating near/at the RCM visibly displaces the tip,
                    # not just rotates it" bug: unlike auto_orient_about_rcm()
                    # (which always publishes v_lin = omega x lever to hold
                    # the tip on the RCM while rotating), this manual path had
                    # no such compensation. Add it here too, pivoting about
                    # the INTERNAL rcm once TUNNELed (matches 'p' REORIENT's
                    # own pivot choice), else the latched (surface) RCM.
                    #
                    # 2026-10-06: rcm_ok() defaults to S.rcm_source=="manual",
                    # which reads _rcm["p"] -- populated from the ROBOT's OWN
                    # continuously-published RCMPoint ROS topic (_cb_rcm),
                    # NOT from anything this script's keys set. That topic is
                    # basically always nonzero for this RCM-constrained robot,
                    # so this pivot-compensation is normally ALWAYS active for
                    # j/k/l/m/u/o -- restarting this script does NOT turn it
                    # off (confirmed live: fresh process, still coupled). That
                    # is correct/safe once actually inserted through a
                    # trocar, but is just unwanted coupling during free-space
                    # work like the 15-point touch calibration (nothing is
                    # inserted yet, there is no real RCM to protect). Added
                    # MANUAL_JOG_RCM_PIVOT=0 to suppress it on request for
                    # exactly that use case; defaults to "1" (old behavior
                    # unchanged) so nothing changes unless explicitly opted
                    # out of.
                    if jog["ang"] is not None and MANUAL_JOG_RCM_PIVOT and rcm_ok():
                        ee_now = _ee["p"]
                        pivot = S.internal_rcm if (S.tunneled and S.internal_rcm is not None) else cur_rcm()
                        if ee_now is not None and pivot is not None:
                            omega = np.array(ang_cmd, float)
                            lever = ee_now - pivot  # EE origin, NOT tip -- see docstring pt.11
                            v_pivot = np.cross(omega, lever)
                            sp = np.linalg.norm(v_pivot)
                            if sp > SPEED_MAX:
                                v_pivot = v_pivot * (SPEED_MAX / sp)
                            lin_cmd = list(v_pivot)
                    pub.publish(Vector3(*lin_cmd))
                    pub_a.publish(Vector3(*ang_cmd))
                else:
                    zero(); jog["lin"] = None; jog["ang"] = None
                continue
            if k in (ord('q'), 27):
                break
            elif k == ord(' '):
                zero(); jog["lin"] = None; jog["ang"] = None; S.msg = "STOP"
            elif k == ord(']'):
                SPEED = min(SPEED * 1.4, SPEED_MAX); ASPEED = min(ASPEED * 1.4, ASPEED_MAX)
                S.msg = "lin=%.1f ang=%.3f" % (SPEED, ASPEED)
            elif k == ord('['):
                SPEED = max(SPEED / 1.4, 0.2); ASPEED = max(ASPEED / 1.4, 0.005)
                S.msg = "lin=%.1f ang=%.3f" % (SPEED, ASPEED)
            elif k == ord('v'):
                S.rec = not S.rec; S.msg = "recording %s -> %s" % ("ON" if S.rec else "OFF", vid_path)
                print("  " + S.msg)
            elif k == ord('h'):
                if cur_tip() is not None:
                    S.home = cur_tip().copy(); S.state = "HOME_SET"; S.approached = False
                    S.oriented = False
                    S.msg = ""
                    print("  HOME latched: [%.2f %.2f %.2f]" % tuple(S.home))
                    # 2026-10-05: each 'h' press is "start of a new trial" for
                    # the ISMR campaign methodology (10+ trials from DIFFERENT
                    # home positions, not 10 consecutive runs from one home) --
                    # bump the trial counter + diversity-check here, exactly
                    # once per press, so every log_precision/log_correspondence
                    # row from now until the next 'h' press is tagged with
                    # this trial's index and starting pose.
                    note_new_trial_home(S.home)
            elif k == ord('c'):
                calibrate_origin_from_click()
            elif k == ord('1'):
                log_mic_calib_touch("calib_pt_1")
            elif k == ord('2'):
                log_mic_calib_touch("calib_pt_2")
            elif k == ord('3'):
                log_mic_calib_touch("calib_pt_3")
            elif k == ord('4'):
                log_mic_calib_touch("calib_pt_4")
            elif k == ord('5'):
                # 2026-09-24: calib_pt_5..8 added on purpose so the microscope
                # extrinsic fit has MORE points than the 6 DOF of the affine
                # model (was exactly 4 points = 8 equations / 6 unknowns =
                # only 2 residual DOF, i.e. the reported "fit residual" was
                # nearly guaranteed to look small regardless of whether the
                # map is actually accurate -- see mic_extrinsic_calib.py's
                # --loo mode and MIC_INDEPENDENT_VERIFICATION_PROTOCOL.md).
                # With 8 points, mic_extrinsic_calib.py --fit --loo can holds
                # each point out in turn and report a HONEST, non-self-
                # referential accuracy number instead.
                log_mic_calib_touch("calib_pt_5")
            elif k == ord('6'):
                log_mic_calib_touch("calib_pt_6")
            elif k == ord('7'):
                log_mic_calib_touch("calib_pt_7")
            elif k == ord('8'):
                log_mic_calib_touch("calib_pt_8")
            # 2026-10-01 (pt.3): calib_pt_9..15 added on request -- the first
            # 8-point fit's LOO RMS (1.5mm) was judged too thin a margin (only
            # 15 residual DOF for a 9-parameter 3x3-affine model) to trust as
            # the noise floor underneath the ring-to-entry-point correction
            # (which needs to detect a bias that may itself be only ~1-2mm).
            #
            # BUG (found live, 2026-10-01): first attempt used uppercase
            # 'A'-'G', reasoning ord('A')=65 != ord('a')=97 so it should be
            # unambiguous. In practice, in THIS cv2/X11 window setup, Shift
            # was not reliably reaching cv2.waitKey() -- pressing 'A' (with
            # Shift held) still came through as plain 'a' and fired the
            # lowercase jog-translate binding, moving the robot by one
            # unintended ~0.6mm pulse (SPEED=3mm/s * PULSE_S=0.2s). Confirmed
            # by the operator: it happened even on the uppercase retry, i.e.
            # this is NOT a one-off missed-shift mistake, Shift-modified
            # letters are just not safe to rely on here.
            #
            # Fix: use unshifted punctuation keys instead (',', '.', '/',
            # ';', apostrophe, '-', '='). These need NO modifier key at all
            # -- there is no "forgot to hold shift" failure mode -- and none
            # of them were bound to anything before this change, so a
            # mis-press can only ever be a no-op (unbound key), never a jog.
            elif k == ord(','):
                log_mic_calib_touch("calib_pt_9")
            elif k == ord('.'):
                log_mic_calib_touch("calib_pt_10")
            elif k == ord('/'):
                log_mic_calib_touch("calib_pt_11")
            elif k == ord(';'):
                log_mic_calib_touch("calib_pt_12")
            elif k == ord("'"):
                log_mic_calib_touch("calib_pt_13")
            elif k == ord('-'):
                log_mic_calib_touch("calib_pt_14")
            elif k == ord('='):
                log_mic_calib_touch("calib_pt_15")
            elif k == ord('9'):
                log_mic_campaign_touch()
            elif k == ord('0'):
                log_ring_offset_sample()
            elif k == ord('`'):
                if (MIC_RCM_ANGLE_SCHEDULE or INSERT_ANGLE_SCHEDULE) and \
                        _trial_state["idx"] != completed_trial_count() + 1:
                    S.msg = ("'`' refused: press 'h' first -- the trial %d target angle / tilt "
                             "schedule is only applied on 'h'" % (completed_trial_count() + 1))
                    print("  !! " + S.msg); continue
                # 2026-10-05: ONE-key correspondence check + mic RCM lock,
                # see try_correspondence_check()'s docstring. On success
                # this sets _vision_rcm exactly like 'y' would (so the
                # invalidate-previous-APPROACH/ORIENT bookkeeping below is
                # identical to 'y's own handler), PLUS logs the
                # microscope-vs-D405 discrepancy that is the actual point
                # of this key.
                if try_correspondence_check() and (S.approached or S.oriented):
                    S.approached = False
                    S.oriented = False
                    print("  (re-locking RCM invalidates the previous APPROACH/ORIENT -- press"
                          " 'g' then 't' again before 'i')")
            elif k == ord('z'):
                if try_lock_vision_rcm() and (S.approached or S.oriented):
                    S.approached = False
                    S.oriented = False
                    print("  (re-locking RCM invalidates the previous APPROACH/ORIENT -- press"
                          " 'g' then 't' again before 'i')")
            elif k == ord('e'):
                if try_lock_microscope_rcm() and (S.approached or S.oriented):
                    S.approached = False
                    S.oriented = False
                    print("  (re-locking RCM invalidates the previous APPROACH/ORIENT -- press"
                          " 'g' then 't' again before 'i')")
            elif k == ord('y'):
                if try_lock_microscope_rcm_live() and (S.approached or S.oriented):
                    S.approached = False
                    S.oriented = False
                    print("  (re-locking RCM invalidates the previous APPROACH/ORIENT -- press"
                          " 'g' then 't' again before 'i')")
            elif k == ord('H'):
                record_handeye_sample()
            elif k == ord('C'):
                clear_origin_correction()
            elif k == 9:
                _hud["help"] = not _hud["help"]
            elif k == ord('V'):
                _present["on"] = not _present["on"]
                cv2.resizeWindow(win, 1280, 720) if _present["on"] else cv2.resizeWindow(win, 960, 720)
                S.msg = "view: %s" % ("demo" if _present["on"] else "debug")
            elif k == ord('M'):
                # 2026-09-28 (pt.21, real bug -- reported symptom: microscope
                # mark identification "takes a long time AND is still wrong"
                # right after opening/restarting this session). Root cause,
                # confirmed live: this session's top-down light is glaring
                # badly enough that mic_perception.locate() often only finds
                # 1 of 2 marks (n_marks<2 -> occluded=True), which stalls the
                # MIC_MARK_SMOOTH_N=6 accumulation for a long time -- and if
                # the accumulation that DID eventually complete happened to
                # land on a marginal/lucky run of frames right as conditions
                # were still bad, the resulting LOCK can be wrong. Because
                # pt.2's freeze-on-lock fix (see update_mic_mark_live()) is
                # intentionally permanent with NO existing way to undo it
                # short of restarting the whole script (a real-robot-motion
                # action needing the physical safety check every time) --
                # add a pure in-memory reset, no robot motion involved, so a
                # bad mic-mark lock can be cleared and re-accumulated from
                # scratch without touching the robot control process at all.
                # Bound to SHIFT+M (capital, code 77) specifically because
                # plain 'm' is already the ANG-jog-negative-X key (real
                # robot angular velocity) -- must never collide with that.
                for _i in (0, 1):
                    _mic_mark_live["hist"][_i] = []
                    _mic_mark_live["smoothed"][_i] = None
                    _mic_mark_live["jitter_px"][_i] = None
                    _mic_mark_live["n"][_i] = 0
                    _mic_mark_live["locked"][_i] = False
                _mic_mark_live["status_msg"] = "microscope mark lock RESET (SHIFT+M) -- re-accumulating"
                # 2026-10-05: also reset the circle-target tracker so SHIFT+M
                # re-arms BOTH modes' locks together -- otherwise a stale
                # frozen _mic_target_live lock would silently survive a
                # SHIFT+M reset in "circle" mode (same bug class as the
                # original per-mark freeze-forever issue this key fixes).
                _mic_target_live["hist"] = []
                _mic_target_live["smoothed"] = None
                _mic_target_live["jitter_px"] = None
                _mic_target_live["n"] = 0
                _mic_target_live["locked"] = False
                _mic_target_live["status_msg"] = "circle-target tracking not started yet"
                S.msg = _mic_mark_live["status_msg"]
                print("  " + S.msg)
            elif k == ord('x'):
                if try_lock_vision_rcm_with_offset() and (S.approached or S.oriented):
                    S.approached = False
                    S.oriented = False
                    print("  (re-locking RCM invalidates the previous APPROACH/ORIENT -- press"
                          " 'g' then 't' again before 'i')")
            elif k == ord('n'):
                if not rcm_ok():
                    S.msg = "RCM not latched -- lock RCM ('z' or manual) before estimating the surface normal"
                    continue
                if NORMAL_MODE == "ring":
                    res_n = estimate_phantom_normal_from_ring(cur_rcm())
                else:
                    frames_n = pipe.wait_for_frames(timeout_ms=1000)
                    frames_n = align.process(frames_n)
                    color_fn = frames_n.get_color_frame()
                    depth_fn = frames_n.get_depth_frame()
                    if not color_fn or not depth_fn:
                        S.msg = "no frame available for normal estimate"; continue
                    img_n = np.asanyarray(color_fn.get_data())
                    res_n = estimate_phantom_normal_from_frame(img_n, depth_fn, cur_rcm())
                if not res_n["ok"]:
                    S.msg = "normal estimate FAILED: %s" % res_n["fail_reason"]
                    print("  !! " + S.msg)
                    continue
                PHANTOM_NORMAL = {"point": res_n["point_base"], "normal": res_n["normal_base"],
                                   "n_points": res_n["n_points"], "rms": res_n["plane_rms_mm"]}
                _trial_state["normal_fresh"] = True
                _nz = np.asarray(res_n["normal_base"], float)
                print("  phantom tilt vs robot base z = %.1f deg" % np.degrees(np.arccos(
                    min(1.0, abs(_nz[2]) / np.linalg.norm(_nz)))))
                try:
                    tmp_n = NORMAL_NPZ + ".tmp%d" % os.getpid()
                    with open(tmp_n, "wb") as f:
                        np.savez(f, point_base=res_n["point_base"], normal_base=res_n["normal_base"],
                                  normal_cam=res_n["normal_cam"], n_points=res_n["n_points"],
                                  plane_rms_mm=res_n["plane_rms_mm"],
                                  centroid_disagree_mm=res_n["centroid_disagree_mm"],
                                  patch_radius_px=NORMAL_PATCH_RADIUS_PX,
                                  n_red_dropped=res_n["n_red_dropped"], t=time.time(),
                                  normal_mode=NORMAL_MODE)
                    os.replace(tmp_n, NORMAL_NPZ)
                except Exception as e:
                    print("  !! phantom_normal.npz artifact write failed (non-fatal): %s" % e)
                # 2026-10-05: ALSO append this estimate to a running history
                # log (NORMAL_NPZ itself only ever keeps the latest one,
                # overwritten every press) -- this is the raw data
                # phantom_normal_repeatability.py needs to answer "how much
                # does pressing 'n' again, at the SAME physical phantom
                # pose, change the estimate" -- i.e. quantify the single
                # biggest NEW error source MIC_RCM_MODE=circle introduces
                # that MIC_RCM_MODE=mark never depended on at all. Purely
                # additive/non-blocking: never raises, never affects
                # control-loop behavior either way.
                try:
                    with open(NORMAL_SAMPLES_LOG, "a") as _fh:
                        _fh.write(json.dumps({
                            "timestamp": time.time(),
                            "normal_base": res_n["normal_base"].tolist(),
                            "point_base": res_n["point_base"].tolist(),
                            "plane_rms_mm": res_n["plane_rms_mm"],
                            "n_points": res_n["n_points"],
                            "mode": NORMAL_MODE,
                            "spread_deg": res_n.get("spread_deg"),
                        }) + "\n")
                except Exception as e:
                    print("  !! %s append failed (non-fatal): %s" % (NORMAL_SAMPLES_LOG, e))
                S.msg = ("PHANTOM NORMAL set n=[%.3f %.3f %.3f] rms=%.2fmm pts=%d "
                         "centroid lat=%.2fmm depth=%+.2fmm (live, in-process)" % (
                             res_n["normal_base"][0], res_n["normal_base"][1], res_n["normal_base"][2],
                             res_n["plane_rms_mm"], res_n["n_points"],
                             res_n["centroid_lateral_mm"], res_n["centroid_depth_mm"]))
                print("  " + S.msg)
                if res_n["fail_reason"]:
                    print("  !! " + res_n["fail_reason"])
            elif k in LIN:
                if S.approached or S.oriented:
                    S.approached = False
                    S.oriented = False
                    print("  (manual jog invalidates the previous APPROACH/ORIENT -- press 'g'"
                          " then 't' again before 'i')")
                jog["lin"] = LIN[k]; jog["ang"] = None; jog["until"] = now + PULSE_S
            elif k in ANG:
                if S.approached or S.oriented:
                    S.approached = False
                    S.oriented = False
                    print("  (manual jog invalidates the previous APPROACH/ORIENT -- press 'g'"
                          " then 't' again before 'i')")
                jog["ang"] = ANG[k]; jog["lin"] = None; jog["until"] = now + PULSE_S
            elif k == ord('g'):
                if not rcm_ok():
                    S.msg = "RCM not latched"; continue
                if not compute_goals():
                    S.msg = "cannot compute goal (tip==RCM?)"; continue
                ok = auto_move(S.hover_goal, "APPROACH")
                S.state = "HOVER" if ok else S.state
                S.approached = ok
                if ok:
                    S.oriented = False  # new hover point -> any earlier in-place ORIENT is stale
            elif k == ord('t'):
                if NEEDLE_AXIS_LOCAL is None:
                    S.msg = "No current needle axis: collect independent shaft data first"; continue
                if PHANTOM_NORMAL is None:
                    S.msg = "no phantom_normal.npz -- run phantom_normal_estimate.py first"; continue
                if not _trial_state["normal_fresh"]:
                    S.msg = ("ORIENT refused: phantom normal not re-estimated since 'h' -- press 'n' "
                             "(the saved normal may be from a previous phantom placement)")
                    print("  !! " + S.msg); continue
                if not rcm_ok():
                    S.msg = "RCM not latched"; continue
                if S.rcm_source != "vision":
                    S.msg = ("ORIENT needs a vision RCM lock ('z') -- the kinematic "
                             "RCMPoint never touched the phantom/camera, so a normal fit "
                             "around it would be meaningless. Once 'z' succeeds this stays "
                             "valid indefinitely (no need to re-press 'z' repeatedly, see "
                             "docstring pt.12) -- press 'z' again only if the phantom/camera "
                             "actually moved.")
                    print("  " + S.msg); continue
                if not S.approached:
                    S.msg = ("ORIENT refused: no fresh APPROACH ('g') on record -- run 'g' first, "
                             "so the follow-on TUNNEL below is a small controlled advance near the "
                             "RCM, not a move across the whole workspace.")
                    print("  !! " + S.msg); continue
                tdir = desired_insertion_dir()
                if tdir is None:
                    S.msg = "cannot compute target direction"; continue
                pivot_now = cur_tip()
                if pivot_now is None:
                    S.msg = "no FrameEE, abort ORIENT"; continue
                # pt.14 (2026-08-18) used compensate=False here; REVERTED in
                # pt.17 (2026-08-23) back to compensate=True (the default) --
                # see auto_orient_about_rcm()'s docstring for the quantified
                # reasoning (compensate=False guarantees a ~18.5mm tip sweep
                # at 30deg on this rig's TIP_D, not a small residual). Once
                # the angle converges, still proceeds STRAIGHT into
                # TUNNEL/INSERT using the tip's REAL (whatever small residual
                # drift the closed-loop trim did not fully cancel) position
                # -- do_tunnel()/compute_goals() always recompute fresh from
                # the current tip, so they can never be aiming at a stale
                # pre-rotation point.
                ok = auto_orient_about_rcm(tdir, "ORIENT(%.0fdeg)" % INSERT_ANGLE_DEG,
                                           pivot_point=pivot_now)
                if not ok:
                    continue
                S.state = "ORIENTED"
                S.oriented = True
                S.msg = "ORIENT reached target angle -- proceeding directly to insertion (pt.14)"
                print("  " + S.msg)
                if not rcm_ok():
                    S.msg = "RCM not latched"; continue
                if not compute_goals():
                    S.msg = "cannot compute goal"; continue
                if not BIPLANAR_MODE:
                    ok2 = auto_move(S.insert_goal, "INSERT")
                    S.state = "INSERTED" if ok2 else S.state
                else:
                    do_tunnel()
            elif k == ord('i'):
                if not S.approached:
                    S.msg = "APPROACH (g) first"; continue
                if not S.oriented:
                    S.msg = "ORIENT (t) first -- TUNNEL must follow the verified angle, not whatever direction the tip happens to be pointing"; continue
                if not rcm_ok():
                    S.msg = "RCM not latched"; continue
                if not compute_goals():
                    S.msg = "cannot compute goal"; continue
                if not BIPLANAR_MODE:
                    ok = auto_move(S.insert_goal, "INSERT")
                    S.state = "INSERTED" if ok else S.state
                    continue
                # TUNNEL is now normally run automatically right after 't'
                # succeeds (pt.14) -- this manual 'i' path stays available
                # to retry TUNNEL alone (e.g. if the auto-chained attempt
                # was stopped with space or failed) without redoing ORIENT.
                do_tunnel()
            elif k == ord('p'):
                if not BIPLANAR_MODE:
                    S.msg = "'p' (PLUNGE) only applies in BIPLANAR_MODE=1"; continue
                if not S.tunneled or S.internal_rcm is None:
                    S.msg = "TUNNEL (i) first"; continue
                if PHANTOM_NORMAL is None:
                    S.msg = "no phantom_normal.npz"; continue
                n = PHANTOM_NORMAL["normal"]; n = n / np.linalg.norm(n)
                perp_dir = -n  # INSERT_ANGLE_DEG=0 direction: straight along the normal, into tissue
                ok = auto_orient_about_rcm(perp_dir, "REORIENT(internal,perp)", pivot_point=S.internal_rcm)
                if not ok:
                    S.msg = "PLUNGE aborted -- re-orient about internal RCM failed"; continue
                plunge_goal = S.internal_rcm + perp_dir * PLUNGE_DEPTH_MM
                ok = auto_move(plunge_goal, "PLUNGE")
                S.state = "PLUNGED" if ok else S.state
                if ok:
                    S.tunneled = False  # require a fresh TUNNEL before plunging again
                    capture_trial_plunge_stills(plunge_goal)
            elif k == ord('b'):
                if S.home is None:
                    S.msg = "no HOME set"; continue
                if S.state == "PLUNGED" and RETRACT_PAUSE_S > 0:
                    # Report: "pauses to allow for cannula stabilization" before
                    # withdrawing the inner blade -- a plain sleep (no motion
                    # commanded) is the safe way to express that here.
                    S.msg = "stabilization pause (%.1fs) before retract..." % RETRACT_PAUSE_S
                    print("  " + S.msg)
                    t_pause_end = time.time() + RETRACT_PAUSE_S
                    while time.time() < t_pause_end:
                        grab_show()
                # 2026-10-06: RETRACT used to go STRAIGHT from however deep
                # the tip currently is (post-PLUNGE, still physically
                # embedded in the phantom) directly to S.home via a single
                # auto_move() -- which commands a straight 3D line from
                # current-tip to S.home, NOT the reverse of the insertion
                # axis. Whenever S.home isn't collinear with that axis (the
                # normal case), the needle drags sideways while still
                # inside the tissue instead of sliding straight back out
                # along its own track -- reported symptom: "the phantom
                # moves together with the tip" during retract. Fix: ALWAYS
                # withdraw straight back along the SAME axis the tip went
                # in on first (clearing the tissue by HOVER_MM, the same
                # margin already used for the pre-insertion hover point),
                # THEN do the (now tissue-free, drag-safe) move to S.home.
                # Two axis sources, most-authoritative first:
                #   1. BIPLANAR_MODE PLUNGE path: the final insertion
                #      segment was forced purely along -PHANTOM_NORMAL (see
                #      REORIENT(internal,perp)/'p' above), pivoting about
                #      S.internal_rcm -- so the exact reverse axis is
                #      +PHANTOM_NORMAL from S.internal_rcm, independent of
                #      whatever oblique angle TUNNEL/hover used earlier.
                #   2. Otherwise, fall back to S.hover_goal if one was
                #      computed by compute_goals() this trial -- by
                #      construction hover_goal = rcm - u*HOVER_MM lies on
                #      the SAME insertion axis `u` that produced
                #      insert_goal, so retracing to it first is still a
                #      pure axial withdraw (covers plain-INSERT/non-
                #      BIPLANAR runs, where there is no internal_rcm).
                # If NEITHER axis is available, there is no safe way to
                # know the insertion track -- fall through to the old
                # direct-to-home behavior unchanged rather than guessing.
                withdraw_goal = None
                if (BIPLANAR_MODE and S.state == "PLUNGED"
                        and S.internal_rcm is not None and PHANTOM_NORMAL is not None):
                    n = PHANTOM_NORMAL["normal"]; n = n / np.linalg.norm(n)
                    withdraw_goal = S.internal_rcm + n * HOVER_MM
                elif S.hover_goal is not None:
                    withdraw_goal = S.hover_goal.copy()
                if withdraw_goal is not None:
                    ok_w = auto_move(withdraw_goal, "RETRACT_AXIAL")
                    if not ok_w:
                        S.msg = ("RETRACT_AXIAL (straight pull-back along the insertion axis, "
                                  "clearing the tissue) failed/stalled -- NOT continuing to HOME "
                                  "to avoid dragging the phantom while the tip may still be "
                                  "embedded; investigate before retrying 'b'")
                        print("  " + S.msg)
                        continue
                    if S.internal_rcm is not None and PHANTOM_NORMAL is not None and RETRACT_CLEAR_MM > HOVER_MM:
                        n = PHANTOM_NORMAL["normal"]; n = n / np.linalg.norm(n)
                        if not auto_move(S.internal_rcm + n * RETRACT_CLEAR_MM, "RETRACT_CLEAR"):
                            S.msg = "RETRACT_CLEAR (axial lift above the globe before HOME) failed -- not continuing"
                            print("  " + S.msg)
                            continue
                ok = auto_move(S.home, "RETRACT")
                S.state = "RETRACTED" if ok else S.state
                S.approached = False
                S.tunneled = False
                S.internal_rcm = None
                if ok:
                    capture_trial_post_retract_stills()
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(5):
            zero(); time.sleep(0.02)
        vw.release()
        pipe.stop()
        cv2.destroyAllWindows()
        print("\nstopped, zero velocity sent.")
        if os.path.exists(vid_path) and os.path.getsize(vid_path) > 0:
            print("demo video saved -> %s" % vid_path)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
