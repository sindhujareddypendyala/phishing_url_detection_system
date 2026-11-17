# Error Analysis & Model Limitations Report

Evaluated on the unseen holdout test set (2,098 URLs across 1,327 registered domains).

## 1. Top 25 False Positives (Safe URLs flagged as Phishing)

| No | Predicted Prob | Registered Domain | URL Preview | Key Error Patterns |
|:---:|:---:|---|---|---|
| 1 | 99.9% | `microsoftonline.com` | `https://login.microsoftonline.com/dc046083-4ce8-437a-bd49-e995b65...` | High entropy path, multiple subdomains, or security keyword |
| 2 | 99.9% | `microsoftonline.com` | `https://login.microsoftonline.com/abe32f68-c72d-420d-b5bd-750c63a...` | High entropy path, multiple subdomains, or security keyword |
| 3 | 99.9% | `microsoftonline.com` | `https://login.microsoftonline.com/9024fef9-acd7-4a9a-98d1-b7225b6...` | High entropy path, multiple subdomains, or security keyword |
| 4 | 99.9% | `microsoftonline.com` | `https://login.microsoftonline.com/aa687de1-52b4-4b91-9b35-794eba4...` | High entropy path, multiple subdomains, or security keyword |
| 5 | 99.8% | `microsoftonline.com` | `https://login.microsoftonline.com/decee90c-ce03-461e-8c21-dd538e1...` | High entropy path, multiple subdomains, or security keyword |
| 6 | 99.4% | `tumblr.com` | `http://disneyfairies2005-2007.tumblr.com...` | High entropy path, multiple subdomains, or security keyword |
| 7 | 98.9% | `tumblr.com` | `https://lancemyboi2k17.tumblr.com/#_=_...` | High entropy path, multiple subdomains, or security keyword |
| 8 | 98.7% | `tumblr.com` | `https://a842612964.tumblr.com/#_=_...` | High entropy path, multiple subdomains, or security keyword |
| 9 | 98.6% | `freebuddymovies.com` | `http://html.freebuddymovies.com/5/79/pics/55933/nude/188_c1848_01...` | High entropy path, multiple subdomains, or security keyword |
| 10 | 97.4% | `whatismyipaddress.com` | `http://whatismyipaddress.com/dnssec...` | High entropy path, multiple subdomains, or security keyword |
| 11 | 97.2% | `cardbenefitservices.com` | `https://www.cardbenefitservices.com/useraccess/login?returnurl=%2...` | High entropy path, multiple subdomains, or security keyword |
| 12 | 96.5% | `tumblr.com` | `https://par1027.tumblr.com/#_=_...` | High entropy path, multiple subdomains, or security keyword |
| 13 | 96.4% | `songmeanings.com` | `http://songmeanings.com/songs/view/3530822107859545054...` | High entropy path, multiple subdomains, or security keyword |
| 14 | 95.1% | `tumblr.com` | `https://browningboy84.tumblr.com/#_=_...` | High entropy path, multiple subdomains, or security keyword |
| 15 | 94.6% | `about.com` | `http://divorcesupport.about.com/od/abusiverelationships/a/pass_ag...` | High entropy path, multiple subdomains, or security keyword |
| 16 | 94.6% | `uptodown.com` | `http://es-file-explorer.en.uptodown.com/android...` | High entropy path, multiple subdomains, or security keyword |
| 17 | 94.4% | `html5rocks.com` | `https://www.html5rocks.com/en/tutorials/file/filesystem...` | High entropy path, multiple subdomains, or security keyword |
| 18 | 93.7% | `amazonaws.com` | `https://topreviews.s3.amazonaws.com/reviews/169/traditional-jewis...` | High entropy path, multiple subdomains, or security keyword |
| 19 | 92.4% | `wikispaces.com` | `https://site-closed.wikispaces.com...` | High entropy path, multiple subdomains, or security keyword |
| 20 | 91.7% | `enacademic.com` | `http://useful_english.enacademic.com/76401/latency_stage...` | High entropy path, multiple subdomains, or security keyword |
| 21 | 91.5% | `supremecourt.gov.np` | `http://supremecourt.gov.np/web...` | High entropy path, multiple subdomains, or security keyword |
| 22 | 91.2% | `us-passport-service-guide.com` | `https://www.us-passport-service-guide.com...` | High entropy path, multiple subdomains, or security keyword |
| 23 | 91.0% | `rmit.edu.au` | `http://researchbank.rmit.edu.au/view/rmit:3755...` | High entropy path, multiple subdomains, or security keyword |
| 24 | 90.6% | `businessghana.com` | `http://www.businessghana.com/site/autos...` | High entropy path, multiple subdomains, or security keyword |
| 25 | 90.5% | `cyb-sys.com` | `http://cyb-sys.com/jietg/upue...` | High entropy path, multiple subdomains, or security keyword |

## 2. Top 25 False Negatives (Phishing URLs flagged as Safe)

| No | Predicted Prob | Registered Domain | URL Preview | Key Error Patterns |
|:---:|:---:|---|---|---|
| 1 | 2.1% | `idgod.ph` | `https://idgod.ph...` | Short clean URL on compromise-aged domain, no overt keywords |
| 2 | 2.4% | `outofprint.cc` | `https://outofprint.cc/download/2018-payroll-calendar-paylocity.pd...` | Short clean URL on compromise-aged domain, no overt keywords |
| 3 | 6.4% | `checkbbsms.com` | `http://www.checkbbsms.com...` | Short clean URL on compromise-aged domain, no overt keywords |
| 4 | 7.2% | `eebilll.com` | `http://xo.eebilll.com...` | Short clean URL on compromise-aged domain, no overt keywords |
| 5 | 7.3% | `linea1s.com` | `http://www.linea1s.com...` | Short clean URL on compromise-aged domain, no overt keywords |
| 6 | 7.3% | `hop.ru` | `http://lvk.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 7 | 7.9% | `hop.ru` | `http://kama.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 8 | 7.9% | `hop.ru` | `http://joki.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 9 | 8.0% | `bitballoon.com` | `http://faceboook.bitballoon.com...` | Short clean URL on compromise-aged domain, no overt keywords |
| 10 | 8.1% | `hop.ru` | `http://adidas.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 11 | 8.7% | `nationwide-com.com` | `http://nationwide-com.com/nationwide-log-in/nationwide-log-in-int...` | Short clean URL on compromise-aged domain, no overt keywords |
| 12 | 9.2% | `hop.ru` | `http://yaboets.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 13 | 9.5% | `hop.ru` | `http://vkotankte.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 14 | 9.6% | `monclernorway.com` | `http://ie.monclernorway.com...` | Short clean URL on compromise-aged domain, no overt keywords |
| 15 | 10.0% | `hop.ru` | `http://vkfeart.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 16 | 10.2% | `dacompsc.com` | `http://www.dacompsc.com/modules/mod_custom/acesso-seguro-painel/a...` | Short clean URL on compromise-aged domain, no overt keywords |
| 17 | 10.2% | `cam.exposed` | `https://es.cam.exposed...` | Short clean URL on compromise-aged domain, no overt keywords |
| 18 | 10.4% | `cam.exposed` | `https://ru.cam.exposed...` | Short clean URL on compromise-aged domain, no overt keywords |
| 19 | 10.7% | `fndtmc.com` | `https://fndtmc.com...` | Short clean URL on compromise-aged domain, no overt keywords |
| 20 | 10.7% | `cam.exposed` | `https://pt.cam.exposed...` | Short clean URL on compromise-aged domain, no overt keywords |
| 21 | 10.9% | `raulbeneitez.cat` | `http://raulbeneitez.cat...` | Short clean URL on compromise-aged domain, no overt keywords |
| 22 | 11.0% | `hop.ru` | `http://faceboook.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 23 | 11.5% | `cam.exposed` | `https://de.cam.exposed...` | Short clean URL on compromise-aged domain, no overt keywords |
| 24 | 12.5% | `hop.ru` | `http://likevsem.hop.ru...` | Short clean URL on compromise-aged domain, no overt keywords |
| 25 | 12.7% | `cam.exposed` | `https://fr.cam.exposed...` | Short clean URL on compromise-aged domain, no overt keywords |

## 3. Systemic Patterns & Limitations Identified

- **False Positive Drivers:** Legitimate technical, blog, and documentation platforms (e.g. WordPress, StackOverflow, academic domains) frequently employ deeply nested paths, hyphens, and security keywords (`login`, `auth`, `token`, `ssl`).
- **False Negative Drivers:** Sophisticated targeted spear-phishing campaigns hosting phishing portals on compromised legitimate WordPress sites or newly registered brand-neutral domains without overt lexical hints in the URL string.
