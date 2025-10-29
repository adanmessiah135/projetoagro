const form = document.getElementById("loginForm");
const errorBox = document.getElementById("error");
const submitBtn = form.querySelector('button[type="submit"]');

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  // limpa erro
  if (errorBox) {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
  }

  // lê valores do form (pelo atributo name)
  const username = form.username?.value.trim() || "";
  const password = form.password?.value || "";

  if (!username || !password) {
    if (errorBox) {
      errorBox.textContent = "Preencha usuário e senha.";
      errorBox.classList.remove("hidden");
    }
    return;
  }

  // UI: trava botão
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = "Entrando...";
  }

  try {
    const body = new URLSearchParams({ username, password }).toString();

    const res = await fetch("/auth", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
      redirect: "follow",
    });

    // se o Flask respondeu com redirect (ex.: redirect(url_for("dashboard")))
    if (res.redirected) {
      // compatível com seu main.js que verifica localStorage
      try { localStorage.setItem("auth", "1"); } catch {}
      window.location.href = res.url; // normalmente /dashboard
      return;
    }

    // alguns backends podem responder 200 + JSON
    if (res.ok) {
      try {
        const data = await res.json();
        if (data?.redirect) {
          localStorage.setItem("auth", "1");
          window.location.href = data.redirect;
          return;
        }
      } catch { /* ignore JSON parse */ }
    }

    // trata erro (401 com JSON {error: "..."} ou texto simples)
    let msg = "Credenciais inválidas.";
    const text = await res.text();
    try {
      const j = JSON.parse(text);
      msg = j.error || msg;
    } catch {
      if (text) msg = text;
    }
    if (errorBox) {
      errorBox.textContent = msg;
      errorBox.classList.remove("hidden");
    }
  } catch (err) {
    if (errorBox) {
      errorBox.textContent = "Falha na comunicação com o servidor.";
      errorBox.classList.remove("hidden");
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = "LOGIN";
    }
  }
});

