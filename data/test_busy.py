import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app.ai.busy_hour_ai import analyze_busy_hours

data = json.load(open('data/trx.json', 'r', encoding='utf-8'))
r = analyze_busy_hours(data['data'], forecast_days=14)

print('\n--- SUMMARY ---')
print(f"Accuracy: {r['model_accuracy']['accuracy_percent']}%")
print(f"R2: {r['model_accuracy']['r2_score']}")
print(f"MAE: {r['model_accuracy']['mae']}")
print(f"RMSE: {r['model_accuracy']['rmse']}")
print(f"MAPE: {r['model_accuracy']['mape_percent']}%")
print(f"Training samples: {r['model_accuracy']['training_samples']}")
print(f"\nDays forecasted: {len(r['daily_forecasts'])}")
print(f"Peak hours total: {r['summary']['total_peak_hours']}")

print('\nTop 5 Peak Hours:')
for p in r['summary']['top_5_peak_hours']:
    print(f"  {p['date']} {p['hour']} - {p['level']} ({p['predicted_trx']} trx)")

print(f"\nBusiest day: {r['summary']['busiest_predicted_day']['date']} "
      f"({r['summary']['busiest_predicted_day']['day_name']}) "
      f"- {r['summary']['busiest_predicted_day']['total_transactions']} trx")

print(f"\n--- Day 1 Hourly Forecast ({r['daily_forecasts'][0]['date']}) ---")
for h in r['daily_forecasts'][0]['hourly_breakdown']:
    prods = ', '.join([f"{p['product_name']}({p['estimated_qty']})" for p in h['predicted_products'][:3]])
    print(f"  {h['hour']} {h['busy_emoji']} {h['busy_level']:6s} | "
          f"trx={h['predicted_transactions']:5.2f} | "
          f"rev=Rp{h['predicted_revenue']:>10,.0f} | "
          f"products: {prods}")

print(f"\n--- Cross Validation ---")
cv = r['model_accuracy'].get('cross_validation', {})
for name, scores in cv.items():
    if isinstance(scores, dict):
        print(f"  {name}: R2={scores.get('mean_r2', 'N/A')} (std={scores.get('std_r2', 'N/A')})")

print(f"\n--- Feature Importance (RF) ---")
fi = r['model_accuracy'].get('feature_importance_rf', {})
for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * int(imp * 50)
    print(f"  {feat:25s} {imp:.4f} {bar}")
