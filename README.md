# Andrew Nwalie | Portfolio

Source for my portfolio site: **https://chigozie9.github.io/portfolio-site/**

Built with [Quarto](https://quarto.org). Project pages cover QA automation, full-stack and data work:

- Warehouse Inventory Manager (Spring Boot + React): [repo](https://github.com/chigozie9/WareHouse)
- Personal Finance Dashboard (Streamlit): [repo](https://github.com/chigozie9/personal-finance-dashboard)
- Loan Approval Model (scikit-learn): `loan-approval.qmd`
- Netflix User Segmentation (K-Means/PCA): `netflix-clustering.qmd`
- Restaurant Review Trends (NLP): `restaurant-reviews.qmd`

## Build locally

```bash
quarto preview          # live preview
quarto publish gh-pages # deploy to GitHub Pages
```

The loan page runs Python when rendered, so it needs `pandas`, `matplotlib`, `seaborn` and `scikit-learn`.
