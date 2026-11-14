# Offline tournament check on saved fork files. jq only; exploratory. Write-up: docs/novelty/rml3_method.md §2.3.
# Run from the repo root, once per core file (D = results/mi300x_2026-11-10/2026-11-10_prereg_v2):
#   jq -c -n --arg beta 0.9 --slurpfile c $D/main_inj1000/<name>_core.json \
#      --slurpfile r $D/repair_inj1000/<name>_repair.json -f docs/novelty/tournament_offline.jq
# Pass a file that holds {} as r when there is no repair file (the beta arms).
#
# Offline tournament check. Inputs: $c[0] = a *_core.json, $r[0] = the matching *_repair.json or {}.
# Emits one row per (event, judge, horizon): which arm a horizon-H pick would choose and what that arm did at the end.
def yv($f; $b): [ $f.horizons[] as $h | ($f.branches[$b].val[($h|tostring)]) as $v
                  | if $v == null then null else ($v - $f.branches["B*"].val[($h|tostring)]) end ];
def wm($a; $lo; $hi): ($a[$lo:$hi] | if length == 0 then null else (add / length) end);

def mkarm($f; $name; $b):
  ($f.branches[$b]) as $br
  | { name: $name, Y: yv($f; $b),
      aligned: ((($br.horizon_shift // 0) == 0) and (($br.wasted_steps // 0) == 0)),
      tl: $br.train_losses };

def undo4($c): [ {"none":"B0","U_m":"U_m@0","U_v":"U_v@0","U_mv":"U_mv@0"} | to_entries[] | mkarm($c; .key; .value) ];

def events($c; $r):
  ($c.horizons) as $hs
  | [ { phase: $hs[0], kind: "real4", hs: $hs, arms: undo4($c) } ]
  + ( if ($r | has("branches"))
      then [ { phase: $hs[0], kind: "real17", hs: $hs,
               arms: ( undo4($c) + [ $r.branches | keys[] | select(. != "B0" and . != "B*") | . as $b | mkarm($r; $b; $b) ] ) } ]
      else [] end )
  + [ ("1","3","10","30") as $k | select($c.branches["Px\($k)"] != null)
      | { phase: $hs[0], kind: "dose\($k)", hs: $hs,
          arms: [ {"none":"Px\($k)","U_m":"Px\($k)|U_m@0","U_v":"Px\($k)|U_v@0","U_mv":"Px\($k)|U_mv@0","cool":"Px\($k)|cool@0"}
                  | to_entries[] | select($c.branches[.value] != null) | mkarm($c; .key; .value) ] } ];

def judge($ev; $beta):
  ($ev.hs) as $hs | ($hs | length) as $n
  | ($ev.arms | map(select(.name == "none"))[0]) as $none
  | ($ev.arms | min_by(.Y[$n-1])) as $best
  | ($ev.arms | max_by(.Y[$n-1])) as $worst
  | [ range(1; $n-1) as $j
      | ($hs[$j] - $hs[0] + 1) as $H
      | ( ($ev.arms | min_by(.Y[$j])) as $p | {judge: "val", H: $H, pick: $p.name, yend: $p.Y[$n-1]} ),
        ( ($ev.arms | min_by(.Y[$j])) as $p
          | (if ($none.Y[$j] - $p.Y[$j]) < 0.001 then $none else $p end) as $q
          | {judge: "val_band", H: $H, pick: $q.name, yend: $q.Y[$n-1]} ),
        ( ($ev.arms | map(select(.aligned)) | min_by(wm(.tl; 1; $H))) as $p
          | {judge: "train_full", H: $H, pick: $p.name, yend: $p.Y[$n-1]} ),
        ( ($ev.arms | map(select(.aligned)) | min_by(wm(.tl; ($H/2|floor); $H))) as $p
          | {judge: "train_late", H: $H, pick: $p.name, yend: $p.Y[$n-1]} )
    ]
  | map(. + { beta1: $beta, phase: $ev.phase, kind: $ev.kind, narms: ($ev.arms | length),
              best: $best.name, ybest: $best.Y[$n-1], worst: $worst.name, yworst: $worst.Y[$n-1], ynone: $none.Y[$n-1] })
  | map(. + { regret: (.yend - .ybest), harm: (.yend - .ynone) });

events($c[0]; $r[0])[] as $ev | judge($ev; $beta)[]
