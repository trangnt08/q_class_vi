def statistic(file):
    with open(file, 'r') as f:
        abbr = 0; desc = 0; enty = 0; human = 0; loc = 0; num = 0
        abb = 0; exp = 0
        animal = 0; body=0; color=0;creative = 0;currency=0
        dis_med=0;event =0;food =0;instrument =0;lang =0;letter=0;other=0
        plant=0;product=0;religion=0
        sport=0
        substance=0
        symbol=0
        technique=0;term=0;vehicle=0;word=0

        definition=0;description=0;manner=0;reason=0
        group=0;ind=0;title=0;description=0
        city=0;country=0;mountain=0;other=0;state=0
        code=0;count=0;date=0;distance=0;money=0;order=0;other=0;period=0;percent=0;speed=0;temp=0;size=0;weight=0

        for line in f:
            label1, p, label2, question = line.split(" ", 3)
            if label1 == 'ABBR':
                abbr +=1
                if label2 == 'abb':
                    abb+=1
                if label2 == 'exp':
                    abb+=1
            if label1 =='DESC':
                desc+=1
            if label1 == 'ENTY':
                enty +=1
                if label2 == 'animal':
                    animal +=1
                if label2 == 'body':
                    body+=1
            if label1 == 'HUM':
                label1 +=1
            if label1 == 'LOC':
                loc +=1
            if label1 == 'NUM':
                num +=1


"abb","exp","animal","body", "color","cremat","currency","dismed","event","food","instru","lang","letter","plant","product","religion","sport","substance","symbol","techmeth","term","veh","word","def","manner","reason","gr","ind","title","city","country","mount","state","code","count","date","dist","money","order","period","perc","speed","temp","size","weight","desc","other",