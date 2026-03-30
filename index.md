---
layout: default
title: Home
---

# Pages

<ul>
  {% for p in site.pages %}
    {% if p.title and p.url != "/" and p.nav %}
      <li><a href="{{ p.url | relative_url }}">{{ p.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>
