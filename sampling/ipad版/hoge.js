window.addEventListener('load', () => {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    
    // --- ボタン・入力の取得 ---
    const downloadFullDataBtn = document.getElementById('downloadFullDataBtn');
    const plotterBtn = document.getElementById('plotterBtn');
    const clearBtn = document.getElementById('clearBtn');
    const intervalInput = document.getElementById('intervalInput');
    const plotterPreview = document.getElementById('plotterPreview');

    // --- 設定項目 ---
    const GRID_SIZE = 8;
    const COORD_LIMIT = 200;
    const PRESSURE_MAX = 8;
    // 停止とみなす時間閾値 (ms)
    const STOP_THRESHOLD = 500;

    let cellSize = 0;
    
    // Canvasのサイズ設定
    const controlsHeight = document.getElementById('controls').offsetHeight;
    const previewHeight = document.getElementById('previewArea').offsetHeight;
    const availableHeight = window.innerHeight - controlsHeight - previewHeight - 40;
    const size = Math.min(window.innerWidth - 20, availableHeight);
    const finalSize = Math.max(size, 300);

    canvas.width = finalSize;
    canvas.height = finalSize;
    cellSize = canvas.width / GRID_SIZE;

    // グリッド描画関数
    function drawGrid() {
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 0.5;
        for (let i = 1; i < GRID_SIZE; i++) {
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, canvas.height);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo(canvas.width, i * cellSize);
            ctx.stroke();
        }
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
    }
    drawGrid();

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let strokeCount = 0;
    
    let drawingData = []; 
    let lastRecordTime = 0; 

    // ----------------------------------------
    // イベントリスナー
    // ----------------------------------------
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    
    downloadFullDataBtn.addEventListener('click', () => downloadCSV(drawingData, 'unpitsu_data_full.csv'));
    
    plotterBtn.addEventListener('click', () => {
        const plotterData = generatePlotterData(drawingData);
        if (plotterData.length === 0) {
            alert("プロッタ用データが生成できませんでした（データ不足）");
            return;
        }
        updatePreview(plotterData);
        downloadPlotterCSV(plotterData, 'unpitsu_plotter_data.csv');
    });

    clearBtn.addEventListener('click', clearCanvas);

    // ----------------------------------------
    // ヘルパー関数
    // ----------------------------------------
    function getCellId(pos) {
        const cellX = Math.floor(pos.x / cellSize);
        const cellY = Math.floor(pos.y / cellSize);
        const validCellX = Math.max(0, Math.min(cellX, GRID_SIZE - 1));
        const validCellY = Math.max(0, Math.min(cellY, GRID_SIZE - 1));
        return (validCellY * GRID_SIZE) + validCellX;
    }
    
    function getCanvasPosition(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }

    // ----------------------------------------
    // 描画とデータ記録
    // ----------------------------------------
    function startDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = true;
        strokeCount++;
        
        const pos = getCanvasPosition(e); 
        const cellId = getCellId(pos); 

        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        recordData(e, 'down', pos, cellId); 
    }

    function draw(e) {
        if (!isDrawing || e.pointerType !== 'pen') return;
        
        const pos = getCanvasPosition(e); 
        const cellId = getCellId(pos); 

        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        const interval = parseInt(intervalInput.value, 10) || 50;
        if (e.timeStamp - lastRecordTime >= interval) {
            recordData(e, 'move', pos, cellId);
        }
    }

    function stopDrawing(e) {
        if (e.pointerType !== 'pen') return;
        if (!isDrawing) return; 

        const pos = getCanvasPosition(e);
        const cellId = getCellId(pos);
        
        recordData(e, 'up', pos, cellId);
        isDrawing = false;
    }

    function recordData(e, eventType, pos, cellId) {
        lastRecordTime = e.timeStamp;
        const canvasSize = canvas.width; 

        const normX = pos.x / canvasSize;
        const normY = pos.y / canvasSize;
        const convertedX = (normX - 1.0) * COORD_LIMIT;
        const convertedY = normY * -COORD_LIMIT;

        let convertedPressure = 0;
        if (eventType === 'down' || eventType === 'move') {
            convertedPressure = e.pressure * PRESSURE_MAX; 
        }
        
        drawingData.push({
            timestamp: e.timeStamp,
            event_type: eventType,
            stroke_id: strokeCount,
            x: convertedX, 
            y: convertedY, 
            raw_x: pos.x, 
            raw_y: pos.y,
            pressure: convertedPressure,
            cell_id: cellId
        });
    }

    // ----------------------------------------
    // ★ プロッタ用データ生成ロジック
    // ----------------------------------------
    function generatePlotterData(rawData) {
        if (rawData.length === 0) return [];

        const plotterCommands = [];
        
        const strokes = {};
        rawData.forEach(d => {
            if (!strokes[d.stroke_id]) strokes[d.stroke_id] = [];
            strokes[d.stroke_id].push(d);
        });

        Object.keys(strokes).sort((a,b)=>a-b).forEach(strokeId => {
            const points = strokes[strokeId];
            if (points.length === 0) return;

            // 各ポイントに「次の点までの時間差」を計算して付与
            for (let i = 0; i < points.length - 1; i++) {
                points[i].duration = points[i+1].timestamp - points[i].timestamp;
            }
            points[points.length - 1].duration = 0;

            // A. ストローク開始点への移動 (A0)
            const startPoint = points[0];
            plotterCommands.push({
                mode: 'A0',
                x: startPoint.x,
                y: startPoint.y,
                z: 0,
                f: '',
                delay: '',
                cellId: startPoint.cell_id + 1 // 1-64
            });

            // B. セルごとのグループ化と代表点選出
            const cellGroups = [];
            let currentGroup = { cellId: points[0].cell_id, points: [points[0]] };
            
            for (let i = 1; i < points.length; i++) {
                const p = points[i];
                if (p.cell_id === currentGroup.cellId) {
                    currentGroup.points.push(p);
                } else {
                    cellGroups.push(currentGroup);
                    currentGroup = { cellId: p.cell_id, points: [p] };
                }
            }
            cellGroups.push(currentGroup);

            // 代表点リスト
            const representativePoints = [];
            
            cellGroups.forEach(group => {
                // 1. グループ内に「停止点」があるか探す
                // 停止点 = 次の点までの時間が閾値を超えている点
                let stopPoint = null;
                let maxDuration = 0;

                group.points.forEach(p => {
                    if (p.duration >= STOP_THRESHOLD) {
                        if (p.duration > maxDuration) {
                            maxDuration = p.duration;
                            stopPoint = p;
                        }
                    }
                });

                if (stopPoint) {
                    // ★ 停止点があれば、中心座標ではなくその座標を優先して採用
                    // プロパティ isStopPoint を付与して後でD1判定に使う
                    stopPoint.isStopPoint = true;
                    stopPoint.waitTime = stopPoint.duration;
                    representativePoints.push(stopPoint);
                } else {
                    // 2. 停止点がなければ、従来通り「セルの中心に最も近い点」を探す
                    const cx = (group.cellId % GRID_SIZE) * cellSize + (cellSize / 2);
                    const cy = Math.floor(group.cellId / GRID_SIZE) * cellSize + (cellSize / 2);

                    let bestPoint = group.points[0];
                    let minDist = Number.MAX_VALUE;

                    group.points.forEach(p => {
                        const dx = p.raw_x - cx;
                        const dy = p.raw_y - cy;
                        const dist = (dx * dx) + (dy * dy);
                        if (dist < minDist) {
                            minDist = dist;
                            bestPoint = p;
                        }
                    });
                    representativePoints.push(bestPoint);
                }
            });

            // C. コマンド生成 (A1 / D1)
            for (let i = 0; i < representativePoints.length - 1; i++) {
                const p1 = representativePoints[i];
                const p2 = representativePoints[i+1];

                // 全体の経過時間
                // (注意: p1が停止点の場合、timestamp差分には停止時間が含まれている)
                let totalTimeDiff = p2.timestamp - p1.timestamp;
                
                // もしp1が停止点なら、ここでD1を出力して待機
                if (p1.isStopPoint) {
                    plotterCommands.push({
                        mode: 'D1',
                        x: p1.x, // 停止した場所
                        y: p1.y,
                        z: p1.pressure,
                        f: '',
                        delay: Math.round(p1.waitTime), // 待機時間
                        cellId: p1.cell_id + 1
                    });
                    
                    // 移動速度計算のために、停止時間を引く
                    totalTimeDiff -= p1.waitTime;
                }

                // 移動時間が極端に短い、または負になる場合は補正
                if (totalTimeDiff <= 0) totalTimeDiff = 10; // 最小値

                // 速度 F の計算
                const feed = (25 / (totalTimeDiff / 1000.0)) * 60;

                plotterCommands.push({
                    mode: 'A1',
                    x: p2.x,
                    y: p2.y,
                    z: p2.pressure,
                    f: Math.round(feed),
                    delay: Math.round(totalTimeDiff), // 移動にかかった時間
                    cellId: p2.cell_id + 1 // 1-64
                });
            }
        });

        return plotterCommands;
    }

    // ----------------------------------------
    // UI更新・CSV出力
    // ----------------------------------------

    function updatePreview(data) {
        const text = data.map(row => {
            const x = row.x.toFixed(2);
            const y = row.y.toFixed(2);
            const z = typeof row.z === 'number' ? row.z.toFixed(2) : '0.00';
            const cell = row.cellId;

            if (row.mode === 'A0') {
                return `${row.mode}, X:${x}, Y:${y}, Z:${z}, Cell:${cell}`;
            } else if (row.mode === 'D1') {
                return `${row.mode}, X:${x}, Y:${y}, Z:${z}, Wait:${row.delay}ms, Cell:${cell}`;
            } else {
                return `${row.mode}, X:${x}, Y:${y}, Z:${z}, F:${row.f} (Move:${row.delay}ms), Cell:${cell}`;
            }
        }).join("\n");
        
        plotterPreview.value = text;
    }

    function downloadPlotterCSV(data, filename) {
        // ヘッダーに Cell_ID を追加
        let csvContent = "Command,X,Y,Z,F,Delay_ms,Cell_ID\n";
        
        csvContent += data.map(row => {
            const x = row.x.toFixed(2);
            const y = row.y.toFixed(2);
            const z = typeof row.z === 'number' ? row.z.toFixed(2) : '0.00';
            const f = row.f !== '' ? row.f : '';
            const d = row.delay !== '' ? row.delay : '';
            const cell = row.cellId;
            
            return `${row.mode},${x},${y},${z},${f},${d},${cell}`;
        }).join("\n");

        const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
        const blob = new Blob([bom, csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function downloadCSV(data, filename) {
        if (data.length === 0) {
            alert("データがありません。");
            return;
        }
        const headers = Object.keys(data[0]).filter(k => k !== 'raw_x' && k !== 'raw_y').join(',');
        const csvRows = data.map(row => {
            return Object.keys(row).filter(k => k !== 'raw_x' && k !== 'raw_y').map(key => {
                const value = row[key];
                if (typeof value === 'number') {
                    if (key === 'x' || key === 'y') return value.toFixed(2);
                    if (key === 'pressure' || key === 'tilt_x' || key === 'tilt_y') return value.toFixed(4);
                }
                return value;
            }).join(',');
        });
        const csvContent = headers + "\n" + csvRows.join("\n");
        const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
        const blob = new Blob([bom, csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function clearCanvas() {
        if (confirm("データを消去しますか？")) {
            drawingData = [];
            strokeCount = 0;
            lastRecordTime = 0;
            plotterPreview.value = ""; 
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGrid(); 
        }
    }
});