"""Deterministic synthetic data generator for the teaching version of the capstone.
All records are fictional. Run: python generate_data.py
"""
import csv, math, os, random
from datetime import datetime, timedelta

rng = random.Random(42)
RAW, POL = "data/raw", "data/policies"
os.makedirs(RAW, exist_ok=True); os.makedirs(POL, exist_ok=True)
CITIES = ["Chennai", "Mumbai", "Bengaluru", "Delhi", "Hyderabad", "Pune"]
PRODUCTS = [
("Basmati Rice 1kg","Grocery",129),("Toor Dal 1kg","Grocery",165),("Atta 5kg","Grocery",245),
("Sunflower Oil 1L","Grocery",155),("Sugar 1kg","Grocery",48),("Salt 1kg","Grocery",24),
("Tea Powder 250g","Grocery",120),("Poha 500g","Grocery",42),("Toned Milk 500ml","Dairy",29),
("Curd 400g","Dairy",45),("Paneer 200g","Dairy",95),("Butter 100g","Dairy",58),
("Cheese Slices 200g","Dairy",130),("Ghee 500ml","Dairy",340),("Eggs 12 Pack","Dairy",84),
("Lassi 200ml","Dairy",25),("Potato Chips 52g","Snacks",20),("Namkeen 200g","Snacks",60),
("Biscuits 150g","Snacks",30),("Dark Chocolate 80g","Snacks",150),("Instant Noodles 4 Pack","Snacks",56),
("Popcorn 100g","Snacks",45),("Peanut Butter 340g","Snacks",210),("Cold Coffee Can 180ml","Snacks",90),
("Shampoo 340ml","Personal Care",280),("Face Wash 100ml","Personal Care",190),
("Toothpaste 150g","Personal Care",98),("Soap 4 Pack","Personal Care",150),
("Hand Wash 250ml","Personal Care",99),("Body Lotion 400ml","Personal Care",350),
("Deodorant 150ml","Personal Care",220),("Sunscreen 50g","Personal Care",399),
("Wireless Earbuds","Electronics",1499),("Power Bank 10000mAh","Electronics",1299),
("USB-C Cable 1m","Electronics",299),("Phone Charger 20W","Electronics",799),
("Smart Bulb","Electronics",599),("Bluetooth Speaker","Electronics",1999),
("Phone Stand","Electronics",349),("Memory Card 64GB","Electronics",549)]
def write_csv(name, header, rows):
    with open(os.path.join(RAW,name),"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)
def poisson(lam):
    L,k,p=math.exp(-lam),0,1.0
    while True:
        p*=rng.random()
        if p<=L: return k
        k+=1
def mangle_city(c):
    r=rng.random()
    return c.lower() if r<.20 else c.upper() if r<.32 else (" "+c+" ") if r<.42 else c
def mangle_gender(g):
    return rng.choice({"male":["male","Male","M","m","MALE"],"female":["female","Female","F","f","FEMALE"]}[g])

write_csv("products.csv",["product_id","product_name","category","price_inr"],
          [[f"P{i+1:03d}",n,c,p] for i,(n,c,p) in enumerate(PRODUCTS)])
n_cust=1200; arche={}; cust=[]
bad_age={10:280,110:-4,210:0,310:150,410:999,510:200}
for i in range(n_cust):
    cid=f"C{i+1:04d}"; r=rng.random()
    arche[cid]="premium" if r<.20 else "frequent" if r<.60 else "rare"
    signup=datetime(2024,1,1)+timedelta(days=rng.randint(0,455))
    age=bad_age.get(i,rng.randint(18,60))
    if i not in bad_age and rng.random()<.05: age=""
    gender="" if rng.random()<.03 else mangle_gender(rng.choice(["male","female"]))
    cust.append([cid,mangle_city(rng.choice(CITIES)),signup.strftime("%Y-%m-%d"),age,gender])
write_csv("customers.csv",["customer_id","city","signup_date","age","gender"],cust)

cat_idx={}
for i,(_,c,_) in enumerate(PRODUCTS): cat_idx.setdefault(c,[]).append(i)
rows=[]; oid=100000; start=datetime(2025,4,1); win_days=183
rates={"premium":1.0,"frequent":3.5,"rare":0.5}
for cid,a in arche.items():
    for _ in range(poisson(rates[a]*6)):
        oid+=1
        if a=="premium":
            pool=cat_idx["Electronics"] if rng.random()<.7 else cat_idx["Personal Care"]+cat_idx["Snacks"]
            pi=rng.choice(pool); qty=rng.randint(1,2)
        elif a=="frequent":
            pi=rng.choice(cat_idx["Grocery"]+cat_idx["Dairy"]+cat_idx["Snacks"]); qty=rng.randint(1,3)
        else:
            pi=rng.choice(cat_idx["Grocery"]+cat_idx["Dairy"]+cat_idx["Snacks"]+cat_idx["Personal Care"]); qty=rng.randint(1,2)
        ts=start+timedelta(days=rng.randint(0,win_days-1),hours=rng.randint(7,22),minutes=rng.randint(0,59),seconds=rng.randint(0,59))
        disc=rng.choice([0,0,0,5,10,15,20,25,40]); status="cancelled" if rng.random()<.10 else "completed"
        amount=round(PRODUCTS[pi][2]*qty*(1-disc/100),2)
        if status=="cancelled": delivery=""
        else:
            delivery=rng.randint(46,75) if rng.random()<.10 else max(8,min(45,int(round(rng.gauss(24,8)))))
            if rng.random()<.06: delivery=""
        rows.append([f"ORD-{oid}",cid,f"P{pi+1:03d}",qty,ts.strftime("%Y-%m-%d %H:%M:%S"),status,disc,amount,delivery])
out=[]
for k,r in enumerate(rows):
    if k%100==7 and r[5]=="completed" and r[8]!="": r[8]=-10
    if r[5]=="completed" and rng.random()<.01: r[7]="N/A" if rng.random()<.5 else "error"
    out.append(list(r))
    if rng.random()<.05:
        early=list(r); early[4]=(datetime.strptime(r[4],"%Y-%m-%d %H:%M:%S")-timedelta(minutes=rng.randint(1,10))).strftime("%Y-%m-%d %H:%M:%S")
        early[5]="pending"; early[8]=""; out.append(early)
    if rng.random()<.01: out.append(list(r))
rng.shuffle(out)
write_csv("orders.csv",["order_id","customer_id","product_id","quantity","order_ts","status","discount_pct","order_amount","delivery_minutes"],out)

T={
"billing":["I was charged twice for order {o}. Please refund the extra payment. My email is {e}.","The invoice for {o} shows a wrong amount. Call me on {p} to correct the bill.","Extra money was deducted for order {o}, please check the bill. Contact {e}."],
"delivery":["My order {o} has not arrived and the rider is not picking up calls. Please call {p}.","Delivery for {o} is very late, it was promised in 20 minutes. Write to {e}.","The rider marked {o} as delivered but I never received the package. Call {p}."],
"technical":["The app crashes every time I open the cart. I tried to place {o} again and got an error. Email {e}.","I cannot login, the OTP never arrives on {p}.","The payment page shows an error and the app freezes when I apply a coupon. Order {o}."],
"fraud":["There is an unauthorized transaction on my card linked to order {o}. I did not place it. Call {p}.","Someone hacked my account and used my wallet balance. This is fraud, contact {e}.","I was charged for order {o} that I never made, it looks like fraud on my account. Call {p}."]}
order_by_cust={}
for r in out: order_by_cust.setdefault(r[1],r[0])
eligible=sorted(order_by_cust); tickets=[]; cats=list(T)
for n in range(1,157):
    cat=cats[(n-1)%4]; cid=rng.choice(eligible)
    txt=rng.choice(T[cat]).format(o=order_by_cust[cid],e=f"user{n}@example.com",p=str(rng.choice("6789"))+"".join(rng.choice("0123456789") for _ in range(9)))
    tickets.append([f"T{n:04d}",cid,order_by_cust[cid],txt,cat])
fill="The same problem keeps happening and nobody has given me a clear answer yet. "
for n in range(157,161): tickets.append([f"T{n:04d}",rng.choice(eligible),"","","none"])
for n in range(161,165): tickets.append([f"T{n:04d}",rng.choice(eligible),"",("My delivery was late. "+fill*40).strip(),"delivery"])
rng.shuffle(tickets)
write_csv("tickets.csv",["ticket_id","customer_id","order_id","text","true_category"],tickets)

DOCS={
"refund_policy.txt":"Refund Policy. A refund can be requested only for orders whose status is completed. Orders with status cancelled were never charged, so no refund applies to them. If a completed order is delivered in more than 45 minutes, the customer receives a full refund of the order amount. If a completed order is delivered in 45 minutes or less, no late-delivery refund is paid. Refunds up to INR 1000 are approved automatically by the support system. A refund above INR 1000 must be approved by a human support lead before money is returned. Approved refunds are credited to the original payment method within 5 working days. A refund request must be raised within 7 days of the delivery date.",
"delivery_policy.txt":"Delivery Policy. Standard delivery is promised within 30 minutes of order confirmation in all supported cities. The supported cities are Chennai, Mumbai, Bengaluru, Delhi, Hyderabad and Pune. A delivery fee of INR 25 is charged on orders below INR 199, and delivery is free on orders of INR 199 or more. During heavy rain the promised time may extend to 60 minutes and the app shows a rain notice. A rider will attempt to contact the customer twice before marking an order as undeliverable. Electronics items are delivered only after the customer shares a one time password with the rider.",
"payment_policy.txt":"Payment Policy. Customers can pay using UPI, debit card, credit card or cash on delivery. Cash on delivery is available only for orders up to INR 2500. If a payment fails but money is deducted, the amount is returned automatically within 48 hours. A customer who sees two charges for one order should raise a billing ticket with the order identifier. Wallet balance can be used for any order and is never refunded as cash. Coupons cannot be combined with wallet balance on the same order.",
"privacy_policy.txt":"Privacy Policy. Customer phone numbers and email addresses are used only for order updates and support. Support tickets are processed with personal details masked before any automated tool reads them. Delivery addresses are deleted from active systems 180 days after the last order. A customer can ask for account deletion and the request is completed within 30 days. Payment card numbers are never stored by the support system.",
"escalation_policy.txt":"Escalation Policy. A ticket about fraud or an unauthorized transaction is marked high priority and sent to the security team within 2 hours. Tickets from premium customers are answered within 1 hour, and all other tickets within 24 hours. If an automated assistant cannot resolve a ticket within 6 steps, it must hand the case to a human agent with a short summary. A human agent must call the customer back if such a high-value request is rejected."}
for fn,body in DOCS.items():
    with open(os.path.join(POL,fn),"w",encoding="utf-8") as f: f.write(body+"\n")
QA=[
("What delivery time makes a completed order eligible for a full refund?","refund_policy.txt","more than 45 minutes"),
("Who must approve a refund above INR 1000?","refund_policy.txt","human support lead"),
("How many days do I have to raise a refund request?","refund_policy.txt","within 7 days"),
("What is the delivery fee on small orders?","delivery_policy.txt","INR 25"),
("On orders of what value is delivery free?","delivery_policy.txt","INR 199 or more"),
("Why does the electronics delivery need a one time password?","delivery_policy.txt","one time password"),
("Is cash on delivery available for a large order?","payment_policy.txt","INR 2500"),
("How soon is money returned when a payment fails but is deducted?","payment_policy.txt","48 hours"),
("Can wallet balance be refunded as cash?","payment_policy.txt","never refunded as cash"),
("When are delivery addresses deleted from active systems?","privacy_policy.txt","180 days"),
("How fast must a fraud ticket reach the security team?","escalation_policy.txt","within 2 hours"),
("After how many steps must an automated assistant hand over to a human?","escalation_policy.txt","6 steps"),
("What is the favourite colour of the company mascot?","NONE",""),
("Who won the cricket world cup final?","NONE","")]
with open(os.path.join(RAW,"policy_eval.csv"),"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["question","expected_doc","expected_fact"]); w.writerows(QA)
# Agent scenarios will be derived from the clean order table by src/agent.py for robustness.
print(f"generated: customers={len(cust)} products={len(PRODUCTS)} orders_raw={len(out)} tickets={len(tickets)}")
