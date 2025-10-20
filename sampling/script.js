window.addEventListener('load', () => {
    const canvas = document.getElementById('drawCanvas');
    const ctx = canvas.getContext('2d');
    
    // --- ボタンの取得 ---
    const downloadFullDataBtn = document.getElementById('downloadFullDataBtn');
    const downloadTransitionBtn = document.getElementById('downloadTransitionBtn');
    const clearBtn = document.getElementById('clearBtn');

    // --- 設定項目 ---
    const GRID_SIZE = 8;
    const COORD_LIMIT = 200;
    const PRESSURE_MAX = 8; // ★筆圧の最大値を8に設定

    let cellSize = 0;
    
    // Canvasのサイズ設定
    const controlsHeight = document.getElementById('controls').offsetHeight;
    const availableHeight = window.innerHeight - controlsHeight;
    const size = Math.min(window.innerWidth - 4, availableHeight - 4);
    
    canvas.width = size;
    canvas.height = size;
    cellSize = canvas.width / GRID_SIZE;

    // グリッド描画
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

    // 描画用の設定
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#000000';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let strokeCount = 0;
    
    let drawingData = []; 
    let cellTransitions = [];
    let lastCellId = -1;

    // ----------------------------------------
    // イベントリスナー
    // ----------------------------------------
    canvas.addEventListener('pointerdown', startDrawing);
    canvas.addEventListener('pointermove', draw);
    canvas.addEventListener('pointerup', stopDrawing);     // ★ 'up' を検出
    canvas.addEventListener('pointerleave', stopDrawing);   // ★ 'leave' を検出
    
    downloadFullDataBtn.addEventListener('click', () => downloadCSV(drawingData, 'unpitsu_data_full.csv'));
    downloadTransitionBtn.addEventListener('click', () => downloadCSV(cellTransitions, 'unpitsu_data_transitions.csv'));
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
    // 描画とデータ記録の関数
    // ----------------------------------------

    function startDrawing(e) {
        if (e.pointerType !== 'pen') return;
        isDrawing = true;
        strokeCount++;
        
        const pos = getCanvasPosition(e); 
        const cellId = getCellId(pos); 
        lastCellId = cellId; 

        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        recordData(e, 'down', pos, cellId); // eventType 'down'
    }

    function draw(e) {
        if (!isDrawing || e.pointerType !== 'pen') return;
        
        const pos = getCanvasPosition(e); 
        const cellId = getCellId(pos); 

        // --- セル移動の検出ロジック (変更なし) ---
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
        // ------------------------------

        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        
        recordData(e, 'move', pos, cellId); // eventType 'move'
    }

    // --- ★修正点：ペンが離れた瞬間の記録 ---
    function stopDrawing(e) {
        if (e.pointerType !== 'pen') return;
        if (!isDrawing) return; // 既に 'up' 処理済みの場合は何もしない

        // ペンが離れた最後の位置でデータを記録
        const pos = getCanvasPosition(e);
        const cellId = getCellId(pos);
        recordData(e, 'up', pos, cellId); // eventType 'up'

        isDrawing = false;
        lastCellId = -1; 
    }

    // --- ★修正点：生データを記録する関数 (筆圧変換) ---
    function recordData(e, eventType, pos, cellId) {
        const canvasSize = canvas.width; 

        // 座標変換 (右上が0, 左下-200)
        const normX = pos.x / canvasSize;
        const normY = pos.y / canvasSize;
        const convertedX = (normX - 1.0) * COORD_LIMIT;
        const convertedY = normY * -COORD_LIMIT;

        // --- ★筆圧の計算ロジック ---
        let convertedPressure = 0;
        if (eventType === 'down' || eventType === 'move') {
            // 'down' または 'move' の場合、e.pressure (0.0-1.0) をスケール変換
            convertedPressure = e.pressure * PRESSURE_MAX; // (0.0 〜 8.0)
        }
        // eventType === 'up' の場合は、デフォルトの 0 が使用される
        // ----------------------------

        drawingData.push({
            timestamp: e.timeStamp,
            event_type: eventType, // 'down', 'move', 'up' のいずれか
            stroke_id: strokeCount,
            x: convertedX, 
            y: convertedY, 
            pressure: convertedPressure, // ★変換後の筆圧
            tilt_x: e.tiltX,
            tilt_y: e.tiltY,
            cell_id: cellId
        });
    }

    // ----------------------------------------
    // CSV処理とクリアの関数
    // ----------------------------------------

    function downloadCSV(data, filename) {
        if (data.length === 0) {
            alert("データがありません。");
            return;
        }

        const headers = Object.keys(data[0]).join(',');
        
        const csvRows = data.map(row => {
            return Object.keys(row).map(key => {
                const value = row[key];
                
                // 数値の丸め処理 (変更なし)
                if (typeof value === 'number') {
                    if (key === 'x' || key === 'y') {
                        return value.toFixed(2); 
                    }
                    if (key === 'pressure' || key === 'tilt_x' || key === 'tilt_y') {
                        // pressure も 0.0000 〜 8.0000 で記録
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
        if (confirm("データを消去しますか？")) {
            drawingData = [];
            cellTransitions = [];
            strokeCount = 0;
            lastCellId = -1;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawGrid(); 
        }
    }
});