/**
 * charts.js — utilitários de renderização Plotly no cliente.
 *
 * As figuras são construídas no servidor (plotly.express) e chegam ao
 * browser já serializadas em JSON (plotly.io.to_json). Aqui apenas as
 * desenhamos com Plotly.newPlot e garantimos responsividade.
 */
(function (global) {
    "use strict";

    /** Configuração comum a todos os gráficos. */
    const DEFAULT_CONFIG = {
        responsive: true,
        displaylogo: false,
        locale: "pt-PT",
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
        toImageButtonOptions: { format: "png", filename: "nba_analytics", scale: 2 },
    };

    /**
     * Renderiza uma figura Plotly num elemento.
     * @param {string} elementId  id do <div> de destino
     * @param {{data: Array, layout: Object}} figure  figura serializada
     * @param {Object} [config]  overrides de configuração
     */
    function renderPlotlyChart(elementId, figure, config = {}) {
        const el = document.getElementById(elementId);
        if (!el) {
            console.warn(`[charts] Elemento #${elementId} não encontrado.`);
            return;
        }
        if (typeof Plotly === "undefined") {
            el.innerHTML = '<p class="text-danger p-3">Não foi possível carregar o Plotly.js (verifique a ligação à internet).</p>';
            return;
        }

        el.innerHTML = ""; // remove o spinner de carregamento
        const layout = Object.assign({ autosize: true }, figure.layout || {});

        Plotly.newPlot(el, figure.data || [], layout, Object.assign({}, DEFAULT_CONFIG, config));
    }

    // Redimensiona todos os gráficos quando a janela muda (debounce 150 ms)
    let resizeTimer;
    window.addEventListener("resize", () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            document.querySelectorAll(".js-plotly-plot").forEach((plot) => Plotly.Plots.resize(plot));
        }, 150);
    });

    // Controla as setas indicadoras de conteúdo para cima e para baixo
    function initScrollIndicators() {
        const upBtn = document.getElementById("scroll-up-btn");
        const downBtn = document.getElementById("scroll-down-btn");
        if (!upBtn || !downBtn) return;

        function updateArrows() {
            const scrollTop = window.scrollY || document.documentElement.scrollTop;
            const windowHeight = window.innerHeight;
            const docHeight = document.documentElement.scrollHeight;

            // Mostra seta para cima se já rolou mais de 100px
            upBtn.classList.toggle("visible", scrollTop > 100);
            // Mostra seta para baixo se ainda falta mais de 80px até ao fim da página
            downBtn.classList.toggle("visible", scrollTop + windowHeight < docHeight - 80);
        }

        upBtn.addEventListener("click", () => {
            window.scrollBy({ top: -window.innerHeight * 0.75, behavior: "smooth" });
        });

        downBtn.addEventListener("click", () => {
            window.scrollBy({ top: window.innerHeight * 0.75, behavior: "smooth" });
        });

        window.addEventListener("scroll", updateArrows, { passive: true });
        window.addEventListener("resize", updateArrows);
        updateArrows();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initScrollIndicators);
    } else {
        initScrollIndicators();
    }

    global.renderPlotlyChart = renderPlotlyChart;
})(window);
