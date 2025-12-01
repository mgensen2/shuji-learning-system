window.addEventListener('load', () => {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    
    // --- ボタンと★カチンコ要素の取得 ---
    const downloadFullDataBtn = document.getElementById('downloadFullDataBtn');
    const downloadTransitionBtn = document.getElementById('downloadTransitionBtn');
    const clearBtn = document.getElementById('clearBtn');
    const clapperSignal = document.getElementById('clapperboard-signal'); // ★追加

    // (中略 ... GRID_SIZE, COORD_LIMIT などの設定はそのまま)
    const GRID_SIZE = 8;
    const COORD_LIMIT = 200;
    const PRESSURE_MAX = 8.0; 
    const SAMPLING_INTERVAL = 50; 
    const PSEUDO_PRESSURE_MAX_WIDTH = 30.0; 

    let cellSize = 0;
    
    // (中略 ... Canvasのサイズ設定、drawGrid() はそのまま)
    const controlsHeight = document.getElementById('controls').offsetHeight;
    const availableHeight = window.innerHeight - controlsHeight;
    const size = Math.min(window.innerWidth - 4, availableHeight - 4);
    
    canvas.width = size;
    canvas.height = size;
    cellSize = canvas.width / GRID_SIZE;
    drawGrid();

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let strokeCount = 0;
    
    let drawingData = []; 
    let cellTransitions = [];
    let lastCellId = -1;
    let lastRecordTime = 0; 

    // (中略 ... getCellId, getCanvasPosition はそのまま)
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
    // イベントリスナー
    // ----------------------------------------
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);
    canvas.addEventListener('pointerleave', stopDrawing);
    
    downloadFullDataBtn.addEventListener('click', () => downloadCSV(drawingData, 'unpitsu_data_full.csv'));
    downloadTransitionBtn.addEventListener('click', () => downloadCSV(cellTransitions, 'unpitsu_data_transitions.csv'));
    clearBtn.addEventListener('click', clearCanvas);


    // ----------------------------------------
    // 描画とデータ記録の関数
    // ----------------------------------------

    function startDrawing(e) {
        if (e.pointerType === 'mouse') return;

        // --- ★カチンコ処理 (最初のストロークの時だけ) ---
        if (strokeCount === 0) {
            triggerClapperboard();
        }
        // -------------------------------------

        isDrawing = true;
        strokeCount++;
        
        const pos = getCanvasPosition(e); 
        const cellId = getCellId(pos); 
        lastCellId = cellId; 

        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        recordData(e, 'down', pos, cellId);
    }

    function draw(e) {
        // (中略 ... draw関数の中身は変更なし)
        if (e.pointerType === 'mouse') return;
        if (!isDrawing) return;
        
        const pos = getCanvasPosition(e); 
        const cellId = getCellId(pos); 

        if (cellId !== lastCellId) {
            const current_x = cellId % GRID_SIZE;
            const current_y = Math.floor(cellId / GRID_SIZE);
            const prev_x = lastCellId % GRID_SIZE;
            const prev_y = Math.floor(lastCellId / GRID_SIZE);
            
            const distance = Math.abs(current_x - prev_x) + Math.abs(current_y - prev_y);
            
            if (distance === 1) { 
                cellTransitions.push({
                    timestamp: e.timeStamp,
                    stroke_id: strokeCount,
                    from_cell: lastCellId,
                    to_cell: cellId
                });
            }
            lastCellId = cellId; 
        }

        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        recordData(e, 'move', pos, cellId);
    }

    function stopDrawing(e) {
        // (中略 ... stopDrawing関数の中身は変更なし)
        if (e.pointerType === 'mouse') return;
        if (!isDrawing) return; 

        const pos = getCanvasPosition(e);
        const cellId = getCellId(pos);
        recordData(e, 'up', pos, cellId); 

        isDrawing = false;
        lastCellId = -1; 
    }

    function recordData(e, eventType, pos, cellId) {
        // (中略 ... recordData関数の中身は変更なし)
        if (eventType === 'move') {
            if (e.timeStamp - lastRecordTime < SAMPLING_INTERVAL) {
                return;
            }
        }
        lastRecordTime = e.timeStamp;

        const canvasSize = canvas.width; 
        const normX = pos.x / canvasSize;
        const normY = pos.y / canvasSize;
        const convertedX = (normX - 1.0) * COORD_LIMIT;
        const convertedY = normY * -COORD_LIMIT;

        let convertedPressure = 0;
        if (eventType === 'down' || eventType === 'move') {
            let rawWidth = e.width || 1.0;
            let normalized = Math.min(1.0, rawWidth / PSEUDO_PRESSURE_MAX_WIDTH);
            convertedPressure = normalized * PRESSURE_MAX;
        }
        
        drawingData.push({
            timestamp: e.timeStamp,
            event_type: eventType,
            stroke_id: strokeCount,
            x: convertedX, 
            y: convertedY, 
            pressure: convertedPressure,
            tilt_x: e.tiltX,
            tilt_y: e.tiltY,
            cell_id: cellId
        });
    }

    // --- ★カチンコ信号を発火させる関数 (追加) ---
    function triggerClapperboard() {
        console.log("CLAP!");
        clapperSignal.style.backgroundColor = '#FFFFFF';
        
        setTimeout(() => {
            clapperSignal.style.backgroundColor = '#000000';
        }, 200); // 0.2秒間だけ点灯
    }

    // ----------------------------------------
    // CSV処理とクリアの関数 (変更なし)
    // ----------------------------------------
    function downloadCSV(data, filename) {
        // (中略 ... downloadCSV関数の中身は変更なし)
        if (data.length === 0) {
            alert("データがありません。");
            return;
        }

        const headers = Object.keys(data[0]).join(',');
        
        const csvRows = data.map(row => {
            return Object.keys(row).map(key => {
                const value = row[key];
                
                if (typeof value === 'number') {
                    if (key === 'x' || key === 'y') {
                        return value.toFixed(2); 
                    }
                    if (key === 'pressure' || key === 'tilt_x' || key === 'tilt_y') {
                        return value.toFixed(4); 
                    }
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
        // (中略 ... clearCanvas関数の中身は変更なし)
        if (confirm("データを消去しますか？")) {
            drawingData = [];
            cellTransitions = [];
            strokeCount = 0;
            lastCellId = -1;
            lastRecordTime = 0; 
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGrid(); 
        }
    }
});