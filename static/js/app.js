/**
 * Frontend Controller for Graph RAG vs PageRank Evaluation Platform
 * Implements direct canvas manipulation, in-memory responsive search, and interactive Cytoscape life-cycle.
 */

let cy = null;
let currentDatasetId = "nlp";
let currentAlgorithmMode = "hybrid_rrf";
let customEdges = null;
let pendingSourceNodeId = null;
let currentGraphNodes = [];
let lastSearchResults = [];

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    setupEventListeners();
    await loadAvailableDatasets();
    await loadDatasetGraph(currentDatasetId);
}

function setupEventListeners() {
    // Dataset Selector Change
    document.getElementById("datasetSelect").addEventListener("change", async (e) => {
        currentDatasetId = e.target.value;
        customEdges = null;
        updateSandboxStatus(false);
        await loadDatasetGraph(currentDatasetId);
    });

    // Algorithm Radio Button Switching - Auto re-executes search for live comparison
    document.querySelectorAll('input[name="algorithmMode"]').forEach((radio) => {
        radio.addEventListener("change", async (e) => {
            currentAlgorithmMode = e.target.value;
            const linearContainer = document.getElementById("linearWeightContainer");
            if (currentAlgorithmMode === "hybrid_linear") {
                linearContainer.classList.remove("hidden");
            } else {
                linearContainer.classList.add("hidden");
            }
        });
    });

    // Linear Hybrid Slider Updates
    const slider = document.getElementById("sliderAlpha");
    slider.addEventListener("input", (e) => {
        const prVal = Math.round(e.target.value * 100);
        const ragVal = 100 - prVal;
        document.getElementById("lblPrWeight").innerText = `${prVal}%`;
        document.getElementById("lblRagWeight").innerText = `${ragVal}%`;
    });

    // Search Form Submission
    document.getElementById("searchForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        await executeSearch();
    });

    // Benchmark Suite Trigger & Close
    document.getElementById("btnOpenBenchmark").addEventListener("click", openBenchmarkModal);
    document.getElementById("btnCloseBenchmark").addEventListener("click", closeBenchmarkModal);

    // Reset to Benchmark Graph
    document.getElementById("btnResetGraph").addEventListener("click", async () => {
        customEdges = null;
        pendingSourceNodeId = null;
        updateSandboxStatus(false);
        await loadDatasetGraph(currentDatasetId);
    });
}

async function loadAvailableDatasets() {
    try {
        const res = await fetch("/api/datasets");
        const datasets = await res.json();
        const select = document.getElementById("datasetSelect");
        select.innerHTML = "";
        for (const [id, name] of Object.entries(datasets)) {
            const opt = document.createElement("option");
            opt.value = id;
            opt.innerText = name;
            select.appendChild(opt);
        }
    } catch (err) {
        console.error("Failed to load datasets:", err);
    }
}

async function loadDatasetGraph(datasetId) {
    try {
        const res = await fetch(`/api/dataset/${datasetId}`);
        const data = await res.json();
        currentGraphNodes = data.nodes;
        renderCytoscapeGraph(data.nodes, data.edges);
    } catch (err) {
        console.error("Error loading graph topology:", err);
    }
}

function renderCytoscapeGraph(nodes, edges) {
    const elements = [];

    nodes.forEach((node) => {
        elements.push({
            group: "nodes",
            data: {
                id: node.id,
                label: node.id,
                title: node.title,
                abstract: node.abstract,
                year: node.year
            }
        });
    });

    edges.forEach((edge) => {
        elements.push({
            group: "edges",
            data: {
                id: `e_${edge.source}_${edge.target}`,
                source: edge.source,
                target: edge.target
            }
        });
    });

    if (cy) {
        cy.destroy();
    }

    cy = cytoscape({
        container: document.getElementById("cy"),
        elements: elements,
        style: [
            {
                selector: "node",
                style: {
                    "label": "data(label)",
                    "color": "#ffffff",
                    "font-size": "12px",
                    "font-weight": "bold",
                    "text-valign": "center",
                    "text-halign": "center",
                    "background-color": "#2563eb",
                    "width": "data(size)",
                    "height": "data(size)",
                    // "transition-property": "width, height, background-color, border-width",
                    // "transition-duration": "0.2s"
                }
            },
            {
                selector: "node.connecting-source",
                style: {
                    "border-width": 4,
                    "border-color": "#a855f7",
                    "border-style": "dashed",
                    "background-color": "#7e22ce"
                }
            },
            {
                selector: "node.top-ranked",
                style: {
                    "border-width": 4,
                    "border-color": "#f59e0b",
                    "background-color": "#10b981"
                }
            },
            {
                selector: "edge",
                style: {
                    "width": 2.5,
                    "line-color": "#cbd5e1",
                    "target-arrow-color": "#94a3b8",
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    "arrow-scale": 1.2
                }
            }
        ],
        layout: {
            name: "cose",
            animate: false,
            nodeDimensionsIncludeLabels: true,
            avoidOverlap: true,
            nodeOverlap: 25,
            idealEdgeLength: 110,
            nodeRepulsion: 500000,
            padding: 25
        }
    });

    attachDirectManipulationEvents();
}

// Direct Interaction: Double-click to start connection, click on target to create edge, click edge to delete
function attachDirectManipulationEvents() {
    const tooltip = document.getElementById("nodeTooltip");

    // Tooltip Hover
    cy.on("mouseover", "node", (evt) => {
        const node = evt.target;
        const renderedPos = node.renderedPosition();
        tooltip.innerHTML = `<strong>${node.data("title")}</strong> (${node.data("year")})<br><span class="text-slate-400 text-[10px]">Double-click to link citation</span>`;
        tooltip.style.left = `${renderedPos.x + 20}px`;
        tooltip.style.top = `${renderedPos.y - 15}px`;
        tooltip.classList.remove("hidden");
    });

    cy.on("mouseout", "node", () => {
        tooltip.classList.add("hidden");
    });

    // Single Click: Inspect Node OR Finish Arrow connection
    cy.on("tap", "node", (evt) => {
        const targetNode = evt.target;
        const targetId = targetNode.id();

        if (pendingSourceNodeId) {
            if (pendingSourceNodeId !== targetId) {
                // Add directed edge instantly
                cy.add({
                    group: "edges",
                    data: {
                        id: `custom_${pendingSourceNodeId}_${targetId}`,
                        source: pendingSourceNodeId,
                        target: targetId
                    }
                });
                captureCustomEdges();
                updateSandboxStatus(true);
            }
            // Clear connecting state
            cy.getElementById(pendingSourceNodeId).removeClass("connecting-source");
            pendingSourceNodeId = null;
        } else {
            displayNodeDetails(targetId);
        }
    });

    // Double-Click Node: Begin Arrow creation
    cy.on("dbltap", "node", (evt) => {
        const node = evt.target;
        if (pendingSourceNodeId) {
            cy.getElementById(pendingSourceNodeId).removeClass("connecting-source");
        }
        pendingSourceNodeId = node.id();
        node.addClass("connecting-source");
    });

    // Click Edge: Directly Delete Citation Edge
    cy.on("tap", "edge", (evt) => {
        const edge = evt.target;
        edge.remove();
        captureCustomEdges();
        updateSandboxStatus(true);
    });

    // Tap background: Cancel pending arrow creation
    cy.on("tap", (evt) => {
        if (evt.target === cy && pendingSourceNodeId) {
            cy.getElementById(pendingSourceNodeId).removeClass("connecting-source");
            pendingSourceNodeId = null;
        }
    });
}

function displayNodeDetails(nodeId) {
    const paper = currentGraphNodes.find((n) => n.id === nodeId);
    if (!paper) return;

    document.getElementById("detailBadge").innerText = `Paper ${paper.id} (${paper.year})`;
    document.getElementById("detailTitle").innerText = paper.title;
    document.getElementById("detailAbstract").innerText = paper.abstract;

    // If search results exist, show detailed normalized + raw score breakdown
    const resultObj = lastSearchResults.find((r) => r.node.id === nodeId);
    if (resultObj) {
        document.getElementById("valPrScore").innerHTML = 
            `${resultObj.pagerank_score.toFixed(4)} <span class="text-slate-400 font-normal text-[11px]">(raw: ${resultObj.raw_pagerank_score.toFixed(5)})</span>`;
        document.getElementById("barPr").style.width = `${resultObj.pagerank_score * 100}%`;

        document.getElementById("valRagScore").innerHTML = 
            `${resultObj.rag_score.toFixed(4)} <span class="text-slate-400 font-normal text-[11px]">(raw: ${resultObj.raw_rag_score.toFixed(4)})</span>`;
        document.getElementById("barRag").style.width = `${resultObj.rag_score * 100}%`;

        document.getElementById("valRrfScore").innerHTML = 
            `${resultObj.rrf_score.toFixed(4)} <span class="text-slate-400 font-normal text-[11px]">(raw: ${resultObj.raw_rrf_score.toFixed(5)})</span>`;
        document.getElementById("barRrf").style.width = `${resultObj.rrf_score * 100}%`;
    }
}

async function executeSearch() {
    const query = document.getElementById("queryInput").value.trim();
    if (!query) return;

    const btn = document.getElementById("btnSubmitSearch");
    btn.disabled = true;
    btn.innerText = "...";

    const alpha = parseFloat(document.getElementById("sliderAlpha").value);

    const payload = {
        dataset_id: currentDatasetId,
        query: query,
        algorithm_mode: currentAlgorithmMode,
        custom_edges: customEdges,
        weight_pagerank: alpha,
        weight_rag: 1.0 - alpha
    };

    try {
        const res = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        handleSearchResponse(data);
    } catch (err) {
        console.error("Search failed:", err);
    } finally {
        btn.disabled = false;
        btn.innerText = "Search";
    }
}

function handleSearchResponse(data) {
    const badge = document.getElementById("llmStatusBadge");
    const citation = document.getElementById("llmPaperCitation");
    const responseBox = document.getElementById("llmResponseContent");

    if (!data.is_relevant) {
        badge.className = "text-xs bg-red-100 text-red-700 font-medium px-2.5 py-0.5 rounded-full";
        badge.innerText = "Rejected (Low Similarity)";
        citation.innerText = "";
        responseBox.innerText = data.message;
        lastSearchResults = [];
        return;
    }

    badge.className = "text-xs bg-emerald-100 text-emerald-700 font-medium px-2.5 py-0.5 rounded-full";
    badge.innerText = "Synthesized";
    
    lastSearchResults = data.results;

    if (data.results.length > 0) {
        const top = data.results[0];
        citation.innerText = `Grounded on Rank #1: ${top.node.title} (${top.node.year})`;
        displayNodeDetails(top.node.id);
    }

    responseBox.innerText = data.llm_response || "No response generated.";
    applyScoresToGraph(data.results);
}

function applyScoresToGraph(results) {
    if (!cy || !results || results.length === 0) return;

    // 1. Remove previous winner highlight from all nodes
    cy.nodes().removeClass("top-ranked");

    // 2. Extract active score strictly based on the selected algorithm mode
    const scoredList = results.map((item) => {
        let activeScore = item.rrf_score;
        if (currentAlgorithmMode === "pagerank_lexical") activeScore = item.pagerank_score;
        else if (currentAlgorithmMode === "rag_semantic") activeScore = item.rag_score;
        else if (currentAlgorithmMode === "hybrid_linear") activeScore = item.hybrid_linear_score;

        return {
            id: item.node.id,
            score: activeScore
        };
    });

    // 3. Find the absolute winner (highest score in active mode)
    scoredList.sort((a, b) => b.score - a.score);
    const trueWinnerId = scoredList[0].id;

    // 4. Update diameters and strictly apply green class ONLY to the true winner
    scoredList.forEach((item) => {
        const cyNode = cy.getElementById(item.id);
        if (!cyNode) return;

        // Dynamic diameter: 38px up to 85px
        const diameter = Math.round(38 + item.score * 47);
        cyNode.style({
            "width": `${diameter}px`,
            "height": `${diameter}px`
        });

        // The node with the largest diameter is GUARANTEED to be green
        if (item.id === trueWinnerId) {
            cyNode.addClass("top-ranked");
        }
    });
}

function captureCustomEdges() {
    customEdges = cy.edges().map((e) => ({
        source: e.data("source"),
        target: e.data("target")
    }));
}

function updateSandboxStatus(isCustom) {
    const status = document.getElementById("sandboxStatus");
    if (isCustom) {
        status.innerText = "Topology: Modified (Sandbox Active)";
        status.className = "font-mono text-purple-600 font-semibold";
    } else {
        status.innerText = "Topology: Default Benchmark";
        status.className = "font-mono text-blue-600";
    }
}

async function openBenchmarkModal() {
    const modal = document.getElementById("benchmarkModal");
    const loading = document.getElementById("benchmarkLoading");
    const tableBody = document.getElementById("benchmarkTableBody");

    modal.classList.remove("hidden");
    loading.classList.remove("hidden");
    tableBody.innerHTML = "";

    try {
        const res = await fetch(`/api/benchmark?dataset_id=${currentDatasetId}`, { method: "POST" });
        const data = await res.json();

        loading.classList.add("hidden");
        data.metrics.forEach((m) => {
            const tr = document.createElement("tr");
            tr.className = "hover:bg-slate-50 transition";
            tr.innerHTML = `
                <td class="p-2.5 font-sans font-medium text-slate-900 border-b">${m.algorithm_name}</td>
                <td class="p-2.5 text-center border-b">${(m.precision_at_1 * 100).toFixed(1)}%</td>
                <td class="p-2.5 text-center border-b font-bold text-blue-600">${m.mrr.toFixed(3)}</td>
                <td class="p-2.5 text-center border-b font-bold text-emerald-600">${m.ndcg_at_3.toFixed(3)}</td>
                <td class="p-2.5 text-center border-b text-slate-500">${m.latency_ms.toFixed(1)} ms</td>
            `;
            tableBody.appendChild(tr);
        });
    } catch (err) {
        console.error("Benchmark failed:", err);
    }
}

function closeBenchmarkModal() {
    document.getElementById("benchmarkModal").classList.add("hidden");
}