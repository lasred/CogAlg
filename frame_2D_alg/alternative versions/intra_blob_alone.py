from collections import deque
from intra_comp_ma import *
from itertools import zip_longest

'''
    2D version of 1st-level algorithm is a combination of frame_blobs, intra_blob, and comp_P: optional raster-to-vector conversion.
    intra_blob recursively evaluates each blob for two forks of extended internal cross-comparison and sub-clustering:
    
    der+: incremental derivation cross-comp in high-variation edge areas of +vg: positive deviation of gradient triggers comp_g, 
    rng+: incremental range cross-comp in low-variation flat areas of +v--vg: positive deviation of negated -vg triggers comp_r.
    Each adds a layer of sub_blobs per blob.  
    Please see diagram: https://github.com/boris-kz/CogAlg/blob/master/frame_2D_alg/Illustrations/intra_blob_2_fork_scheme.png
    
    Blob structure, for all layers of blob hierarchy:
    
    root_dert__,  
    Dert = I, iDy, iDx, G, Dy, Dx, M, S (area), Ly (vertical dimension)
    # I: input, (iDy, iDx): angle of input gradient, G: gradient, (Dy, Dx): vertical and lateral Ds, M: match  
    sign, 
    box,  # y0, yn, x0, xn
    dert__,  # box of derts, each = i, idy, idx, g, dy, dx, m
    stack_[ stack_params, Py_ [(P_params, dert_)]]: refs down blob formation tree, in vertical (horizontal) order
    
    # next-fork params:
    fcr, # flag comp rng, also clustering criterion in dert and Dert: g in der+ fork, i+m in rng+ fork? 
    fig, # flag input is gradient
    rdn, # redundancy to higher layers
    rng, # comp range
    sub_layers  # [sub_blobs ]: list of layers across sub_blob derivation tree
                # deeper layers are nested, multiple forks: no single set of fork params?
'''
# filters, All *= rdn:

ave  = 50  # fixed cost per dert, from average m, reflects blob definition cost, may be different for comp_a?
aveB = 1  # fixed cost per intra_blob comp and clustering

# --------------------------------------------------------------------------------------------------------------
# functions, ALL WORK-IN-PROGRESS:

def extend_dert(blob):  # extend dert borders (+1 dert to boundaries)

    y0, yn, x0, xn = blob['box']  # extend dert box:
    _, rY, rX = blob['root_dert__'].shape  # higher dert size
    cP, cY, cX = blob['dert__'].shape  # current dert size

    y0e = y0 - 1; yne = yn + 1; x0e = x0 - 1; xne = xn + 1

    # prevent boundary <0 or >image size:
    if y0e < 0:  y0e = 0; ystart = 0
    else:        ystart = 1
    if yne > rY: yne = rY; yend = ystart + cY
    else:        yend = ystart + cY
    if x0e < 0:  x0e = 0; xstart = 0
    else:        xstart = 1
    if xne > rX: xne = rX; xend = xstart + cX
    else:        xend = xstart + cX

    ini_dert = blob['root_dert__'][:, y0e:yne, x0e:xne]  # extended dert where boundary is masked

    ext_dert__ = ma.array(np.zeros((cP, ini_dert.shape[1], ini_dert.shape[2])))
    ext_dert__[0, ystart:yend, xstart:xend] = blob['dert__'][0].copy() # update i
    ext_dert__[3, ystart:yend, xstart:xend] = blob['dert__'][1].copy() # update g
    ext_dert__[4, ystart:yend, xstart:xend] = blob['dert__'][2].copy() # update dy
    ext_dert__[5, ystart:yend, xstart:xend] = blob['dert__'][3].copy() # update dx
    ext_dert__.mask = blob['dert__'].mask   # set all masks to blob dert mask

    return ext_dert__


def intra_blob(blob, rdn, rng, fig, fcr):  # recursive input rng+ | der+ cross-comp within blob

    # fig: flag input is g | p, fcr: flag comp over rng+ | der+

    spliced_layers = []  # to extend root_blob sub_layers
    ext_dert__ = extend_dert(blob)

    if fcr: dert__ = comp_r(ext_dert__, fig, fcr)  # -> m sub_blobs
    else:   dert__ = comp_g(ext_dert__)  # -> g sub_blobs:

    if dert__.shape[1] >2 and dert__.shape[2] >2:  # min size in y and x

        sub_blobs = cluster_derts(blob, dert__, ave*rdn, fcr, fig)

        blob.update({'fcr': fcr, 'fig': fig, 'rdn': rdn, 'rng': rng,  # fork params
                     'Ls': len(sub_blobs),  # for visibility and next-fork rdn
                     'sub_layers': [sub_blobs] })  # 1st layer of sub_blobs

        for sub_blob in sub_blobs:  # evaluate for intra_blob comp_g | comp_r:
            if sub_blob['sign']:
                if sub_blob['Dert']['M'] > aveB * rdn:  # -> comp_r:
                    blob['sub_layers'] += \
                        intra_blob(sub_blob, rdn + 1 + 1 / blob['Ls'], rng*2, fig=fig, fcr=1)

            elif sub_blob['Dert']['G'] > aveB * rdn:  # -> comp_g
                blob['sub_layers'] += \
                    intra_blob(sub_blob, rdn + 1 + 1 / blob['Ls'], rng=rng, fig=1, fcr=0)

        spliced_layers = [spliced_layers + sub_layers for spliced_layers, sub_layers in
                          zip_longest(spliced_layers, blob['sub_layers'], fillvalue=[])
                          ]
    return spliced_layers


def cluster_derts(blob, dert__, Ave, fcr, fig):  # analog of frame_to_blobs

    if fcr:  # comp_r output;  form clustering criterion:
        if fig: crit__ = dert__[0] + dert__[4] - Ave  # eval by i + m, accum in rng; dert__[:,:,0] if not transposed
        else:   crit__ = Ave - dert__[1]  # eval by -g, accum in rng
    else:    # comp_g output
        crit__ = dert__[4] - Ave  # comp_g output eval by m, or clustering is always by m?

    dert__ = ma.transpose(dert__, axes=(1, 2, 0))  # transpose dert__ into shape [y,x,params]
    stack_ = deque()  # buffer of running vertical stacks of Ps

    for y in range(dert__.shape[0]):  # in height, first and last row are discarded;  print(f'Processing intra line {y}...')

        P_ = form_P_(dert__[y, :], crit__[y, :])  # horizontal clustering, adds a row of Ps
        P_ = scan_P_(P_, stack_, blob['dert__'])  # vertical clustering, adds up_connects per P and down_connect_cnt per stack
        stack_ = form_stack_(P_, blob['dert__'], y)

    sub_blobs =[]  # from form_blob:

    while stack_:  # frame ends, last-line stacks are merged into their blobs:
        sub_blobs.append ( form_blob(stack_.popleft(), blob['dert__']))

    return sub_blobs


# clustering functions:
#-------------------------------------------------------------------------------------------------------------------

def form_P_(dert_, crit_):  # segment dert__ into P__, in horizontal ) vertical order

    P_ = deque()  # row of Ps
    mask_ = dert_[:,0].mask
    sign_ = crit_ > 0
    x0 = 0
    for x in range(len(dert_)):
        if ~mask_[x]:
            x0 = x  # coordinate of first unmasked dert in line
            break
    I, iDy, iDx, G, Dy, Dx, M, L = *dert_[x0], 1  # initialize P params

    _sign = sign_[x0]
    _mask = False  # mask bit per dert

    for x in range(x0+1, dert_.shape[0]):  # loop left to right in each row of derts
        sign = sign_[x]
        mask = mask_[x]
        if (~_mask and mask) or sign != _sign:
            # (P exists and input is not in blob) or sign changed, terminate and pack P:
            P = dict(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L, x0=x0, sign=_sign)
            P_.append(P)
            # initialize P params:
            I, iDy, iDx, G, Dy, Dx, M, L, x0 = 0, 0, 0, 0, 0, 0, 0, 0, x

        if ~mask:  # accumulate P params:
            I += dert_[x][0]
            iDy += dert_[x][1]
            iDx += dert_[x][2]
            G += dert_[x][3]
            Dy += dert_[x][4]
            Dx += dert_[x][5]
            M += dert_[x][6]
            L += 1
            _sign = sign  # prior sign
        _mask = mask

    # terminate and pack last P in a row
    P = dict(I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, L=L, x0=x0, sign=_sign)
    P_.append(P)

    return P_


def scan_P_(P_, stack_, root_dert__):  # merge P into higher-row stack of Ps with same sign and x_coord overlap

    next_P_ = deque()  # to recycle P + up_connect_ that finished scanning _P, will be converted into next_stack_

    if P_ and stack_:  # if both input row and higher row have any Ps / _Ps left

        P = P_.popleft()          # load left-most (lowest-x) input-row P
        stack = stack_.popleft()  # higher-row stacks
        _P = stack['Py_'][-1]     # last element of each stack is higher-row P
        up_connect_ = []          # list of same-sign x-overlapping _Ps per P

        while True:  # while both P_ and stack_ are not empty

            x0 = P['x0']         # first x in P
            xn = x0 + P['L']     # first x beyond P
            _x0 = _P['x0']       # first x in _P
            _xn = _x0 + _P['L']  # first x beyond _P

            if (P['sign'] == stack['sign']
                    and _x0 < xn and x0 < _xn):  # test for sign match and x overlap between loaded P and _P
                stack['down_connect_cnt'] += 1
                up_connect_.append(stack)  # P-connected higher-row stacks are buffered into up_connect_ per P

            if xn < _xn:  # _P overlaps next P in P_
                next_P_.append((P, up_connect_))  # recycle _P for the next run of scan_P_
                up_connect_ = []
                if P_:
                    P = P_.popleft()  # load next P
                else:  # terminate loop
                    if stack['down_connect_cnt'] != 1:  # terminate stack, merge it into up_connects' blobs
                        form_blob(stack, root_dert__)
                    break
            else:  # no next-P overlap
                if stack['down_connect_cnt'] != 1:  # terminate stack, merge it into up_connects' blobs
                    form_blob(stack, root_dert__)
                if stack_:  # load stack with next _P
                    stack = stack_.popleft()
                    _P = stack['Py_'][-1]
                else:  # no stack left: terminate loop
                    next_P_.append((P, up_connect_))
                    break

    while P_:  # terminate Ps and stacks that continue at row's end
        next_P_.append((P_.popleft(), []))  # no up_connect
    while stack_:
        form_blob(stack_.popleft(), root_dert__)  # down_connect_cnt always == 0

    return next_P_  # each element is P + up_connect_ refs


def form_stack_(P_, root_dert__, y):  # Convert or merge every P into its stack of Ps, merge blobs

    next_stack_ = deque()  # converted to stack_ in the next run of scan_P_

    while P_:
        P, up_connect_ = P_.popleft()
        s = P.pop('sign')
        I, G, Dy, Dx, M, iDy, iDx, L, x0 = P.values()
        xn = x0 + L  # next-P x0
        if not up_connect_:
            # initialize new stack for each input-row P that has no connections in higher row:
            blob = dict(Dert=dict(I=0, G=0, Dy=0, Dx=0, M=0, iDy=0, iDx=0, S=0, Ly=0),
                        box=[y, x0, xn], stack_=[], sign=s, open_stacks=1)
            new_stack = dict(I=I, G=G, Dy=0, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=L, Ly=1,
                             y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
            blob['stack_'].append(new_stack)
        else:
            if len(up_connect_) == 1 and up_connect_[0]['down_connect_cnt'] == 1:
                # P has one up_connect and that up_connect has one down_connect=P: merge P into up_connect stack:
                new_stack = up_connect_[0]
                accum_Dert(new_stack, I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=L, Ly=1)
                new_stack['Py_'].append(P)  # Py_: vertical buffer of Ps
                new_stack['down_connect_cnt'] = 0  # reset down_connect_cnt
                blob = new_stack['blob']

            else:  # if > 1 up_connects, or 1 up_connect that has > 1 down_connect_cnt:
                blob = up_connect_[0]['blob']
                # initialize new_stack with up_connect blob:
                new_stack = dict(I=I, G=G, Dy=0, Dx=Dx, M=M, iDy=iDy, iDx=iDx,S=L, Ly=1,
                                 y0=y, Py_=[P], blob=blob, down_connect_cnt=0, sign=s)
                blob['stack_'].append(new_stack)

                if len(up_connect_) > 1:  # merge blobs of all up_connects
                    if up_connect_[0]['down_connect_cnt'] == 1:  # up_connect is not terminated
                        form_blob(up_connect_[0], root_dert__)  # merge stack of 1st up_connect into its blob

                    for up_connect in up_connect_[1:len(up_connect_)]:  # merge blobs of other up_connects into blob of 1st up_connect
                        if up_connect['down_connect_cnt'] == 1:
                            form_blob(up_connect, root_dert__)

                        if not up_connect['blob'] is blob:
                            Dert, box, stack_, s, open_stacks = up_connect['blob'].values()  # merged blob
                            I, G, Dy, Dx, M, iDy, iDx, S, Ly = Dert.values()
                            accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=S, Ly=Ly)
                            blob['open_stacks'] += open_stacks
                            blob['box'][0] = min(blob['box'][0], box[0])  # extend box y0
                            blob['box'][1] = min(blob['box'][1], box[1])  # extend box x0
                            blob['box'][2] = max(blob['box'][2], box[2])  # extend box xn
                            for stack in stack_:
                                if not stack is up_connect:
                                    stack[
                                        'blob'] = blob  # blobs in other up_connects are references to blob in the first up_connect.
                                    blob['stack_'].append(stack)  # buffer of merged root stacks.
                            up_connect['blob'] = blob
                            blob['stack_'].append(up_connect)
                        blob['open_stacks'] -= 1  # overlap with merged blob.

        blob['box'][1] = min(blob['box'][1], x0)  # extend box x0
        blob['box'][2] = max(blob['box'][2], xn)  # extend box xn
        next_stack_.append(new_stack)

    return next_stack_


def form_blob(stack, root_dert__):  # increment blob with terminated stack, check for blob termination

    I, G, Dy, Dx, M, iDy, iDx, S, Ly, y0, Py_, blob, down_connect_cnt, sign = stack.values()
    accum_Dert(blob['Dert'], I=I, G=G, Dy=Dy, Dx=Dx, M=M, iDy=iDy, iDx=iDx, S=S, Ly=Ly)
    # terminated stack is merged into continued or initialized blob (all connected stacks):

    blob['open_stacks'] += down_connect_cnt - 1  # incomplete stack cnt + terminated stack down_connect_cnt - 1: stack itself
    # open stacks contain Ps of a current row and may be extended with new x-overlapping Ps in next run of scan_P_

    if blob['open_stacks'] == 0:  # if number of incomplete stacks == 0, blob is terminated:
        last_stack = stack

        Dert, [y0, x0, xn], stack_, s, open_stacks = blob.values()
        yn = last_stack['y0'] + last_stack['Ly']

        mask = np.ones((yn - y0, xn - x0), dtype=bool)  # mask box, then unmask Ps:
        for stack in stack_:
            stack.pop('sign')
            stack.pop('down_connect_cnt')
            for y, P in enumerate(stack['Py_'], start=stack['y0'] - y0):
                x_start = P['x0'] - x0
                x_stop = x_start + P['L']
                mask[y, x_start:x_stop] = False

        dert__ = (root_dert__[:,y0:yn, x0:xn]).copy()  # copy mask as dert.mask
        dert__.mask = True
        dert__.mask = mask  # overwrite default mask 0s
        root_dert__[:,y0:yn, x0:xn] = dert__.copy()  # assign mask back to blob root dert__

        blob.pop('open_stacks')
        blob.update( root_dert__=root_dert__, box=(y0, yn, x0, xn), dert__=dert__)

    return blob


def accum_Dert(Dert: dict, **params) -> None:
    Dert.update({param: Dert[param] + value for param, value in params.items()})