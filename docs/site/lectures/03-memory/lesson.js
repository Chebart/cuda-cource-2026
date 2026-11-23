(() => {
  'use strict';

  const tip = document.getElementById('line-tip');
  const tipTitle = tip.querySelector('strong');
  const tipFields = Array.from(tip.querySelectorAll('[data-field]'), field => ({
    field, key: field.dataset.field, label: field.previousElementSibling.textContent
  }));
  let visible = null;
  let pinned = null;
  let hoverTimer;

  function positionTip() {
    if (!visible) return;
    const row = visible.closest('.source-line').getBoundingClientRect();
    if (row.bottom <= 0 || row.top >= innerHeight) {
      tip.hidden = true;
      return;
    }
    tip.hidden = false;
    const {width, height} = tip.getBoundingClientRect();
    const gap = 10;
    const canFitLeft = row.left >= width + 2 * gap;
    const left = canFitLeft ? row.left - width - gap : row.left + gap;
    const top = canFitLeft ? row.top : row.top >= height + 2 * gap ?
      row.top - height - gap : row.bottom + gap;
    tip.style.left = `${Math.max(gap, Math.min(left, innerWidth - width - gap))}px`;
    tip.style.top = `${Math.max(gap, Math.min(top, innerHeight - height - gap))}px`;
  }

  function hideTip() {
    clearTimeout(hoverTimer);
    if (visible) visible.closest('.source-line').classList.remove('hint-active');
    visible = null;
    tip.hidden = true;
  }

  function showTip(button) {
    hideTip();
    visible = button;
    const row = button.closest('.source-line');
    row.classList.add('hint-active');
    tipTitle.textContent = `Строка ${button.textContent}`;
    for (const {field, key} of tipFields) field.textContent = row.dataset[key];
    positionTip();
  }

  function clearPin() {
    if (pinned) pinned.setAttribute('aria-pressed', 'false');
    pinned = null;
  }

  for (const listing of document.querySelectorAll('.code-frame')) {
    const help = listing.querySelector('.hint-how');
    help.hidden = false;
    const group = [];
    for (const row of listing.querySelectorAll('[data-what]')) {
      const number = row.querySelector('.line-no');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'line-no';
      button.dataset.lineNote = '';
      button.textContent = number.textContent;
      button.tabIndex = group.length === 0 ? 0 : -1;
      button.setAttribute('aria-label', `Пояснение к строке ${number.textContent}`);
      button.setAttribute('aria-pressed', 'false');
      const description = document.createElement('span');
      description.className = 'sr-only';
      description.id = `${listing.id}-note-${number.textContent}`;
      description.textContent = tipFields.map(({key, label}) => `${label}: ${row.dataset[key]}`).join(' ');
      button.setAttribute('aria-describedby', description.id);
      listing.append(description);
      number.replaceWith(button);
      group.push(button);
    }
    const select = button => group.forEach(item => { item.tabIndex = item === button ? 0 : -1; });
    for (const [index, button] of group.entries()) {
      button.addEventListener('pointerenter', event => {
        if (event.pointerType !== 'touch') showTip(button);
      });
      button.addEventListener('pointerleave', () => {
        if (visible !== button || pinned === button || button.matches(':focus-visible')) return;
        hoverTimer = setTimeout(() => pinned ? showTip(pinned) : hideTip(), 160);
      });
      button.addEventListener('focus', () => {
        select(button);
        if (button.matches(':focus-visible')) showTip(button);
      });
      button.addEventListener('blur', () => {
        if (pinned === button) clearPin();
        if (visible === button) hideTip();
      });
      button.addEventListener('click', () => {
        if (pinned === button) {
          clearPin();
          hideTip();
          return;
        }
        clearPin();
        pinned = button;
        button.setAttribute('aria-pressed', 'true');
        showTip(button);
      });
      button.addEventListener('keydown', event => {
        const dest = {ArrowUp: index - 1, ArrowDown: index + 1, Home: 0, End: group.length - 1}[event.key];
        if (dest === undefined) return;
        event.preventDefault();
        group[Math.max(0, Math.min(group.length - 1, dest))].focus();
      });
    }
  }

  tip.addEventListener('pointerenter', () => clearTimeout(hoverTimer));
  tip.addEventListener('pointerleave', event => {
    if (!pinned && !visible?.matches(':focus-visible') && !event.relatedTarget?.closest?.('[data-line-note]')) hideTip();
  });
  document.addEventListener('pointerdown', event => {
    if (!event.target.closest('[data-line-note]') && !tip.contains(event.target)) {
      clearPin();
      hideTip();
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && visible) {
      clearPin();
      hideTip();
    }
  });
  window.addEventListener('resize', positionTip);
  window.addEventListener('scroll', positionTip, {passive: true});

  for (const button of document.querySelectorAll('[data-copy]')) {
    const listing = document.getElementById(button.dataset.copy);
    const status = button.parentElement.querySelector('[role="status"]');
    button.hidden = false;
    button.addEventListener('click', async () => {
      const source = Array.from(listing.querySelectorAll('.src'), line => line.textContent).join('\n') + '\n';
      try {
        await navigator.clipboard.writeText(source);
        button.textContent = status.textContent = 'Код скопирован';
      } catch {
        button.textContent = status.textContent = 'Не получилось скопировать. Скачай .cu';
      }
    });
  }
})();
