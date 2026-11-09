CANDIDATES = (
    [
        {"bk": bk, "stages": st, "group_m": gm}
        for bk, sts in ((32, (3, 4)), (64, (2, 3)))
        for st in sts
        for gm in (4, 8, 16)
    ]
    + [
        {"bk": 32, "stages": st, "group_m": gm, "min_blocks": 2}
        for st in (2, 3)
        for gm in (8, 16)
    ]
    + [
        {"bn": 256, "bk": bk, "stages": st, "group_m": gm}
        for bk, st in ((32, 3), (32, 4), (64, 2))
        for gm in (4, 8, 16)
    ]
    + [
        {"bn": bn, "bk": bk, "stages": st, "group_m": gm, "min_blocks": mb, "frag_pipe": fp, "tma": tma}
        for tma, fp in ((False, True), (True, False), (True, True))
        for bn, bk, st, mb in (
            (128, 32, 2, 2),
            (128, 32, 3, 2),
            (128, 32, 3, 1),
            (128, 64, 2, 1),
            (256, 32, 3, 1),
            (256, 32, 4, 1),
            (256, 64, 2, 1),
        )
        for gm in (8, 16)
    ]
    + [
        {
            "bn": bn,
            "bk": bk,
            "stages": st,
            "group_m": 8,
            "frag_pipe": fp,
            "tma": True,
            "warp_spec": True,
            "persistent": pers,
            "setmaxnreg": smr,
        }
        for pers in (False, True)
        for fp in (False, True)
        for bn, bk, st, smr in (
            (128, 32, 4, False),
            (128, 64, 3, False),
            (256, 32, 3, False),
            (256, 32, 3, True),
            (256, 32, 4, True),
            (256, 64, 2, True),
        )
    ]
)


FP8_CANDIDATES = [
    {"bn": bn, "bk": bk, "stages": st, "group_m": gm, "min_blocks": mb, "frag_pipe": fp}
    for bn, bk, st, mb in (
        (128, 64, 3, 2),
        (128, 64, 4, 1),
        (128, 128, 2, 1),
        (256, 64, 3, 1),
        (256, 128, 2, 1),
    )
    for fp in (False, True)
    for gm in (8, 16)
] + [
    {
        "bn": bn,
        "bk": bk,
        "stages": st,
        "group_m": 8,
        "frag_pipe": fp,
        "tma": True,
        "warp_spec": ws,
        "persistent": ws,
        "setmaxnreg": smr,
    }
    for ws in (False, True)
    for fp in (False, True)
    for bn, bk, st, smr in (
        (128, 64, 4, False),
        (128, 128, 3, False),
        (256, 64, 4, True),
        (256, 64, 3, True),
        (256, 128, 2, True),
    )
    for smr in ((smr,) if ws else (False,))
]
