import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from app.api.services.stock_service import get_all_products_summary
data = json.load(open('data/trx.json', 'r', encoding='utf-8'))
r = get_all_products_summary(data['data'])
for x in r:
    print(f"P{x['product_id']} {x['product_name']:20s} | risk={x['risk']:8s} | point={x['risk_point']} | days={x['days_until_empty']}")
