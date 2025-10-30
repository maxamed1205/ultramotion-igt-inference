/**
 * ws_manager.js - WebSocket Manager
 * =================================
 * Gestionnaire de connexions WebSocket pour UltraMotion IGT Dashboard
 * 
 * Responsabilités:
 * - Connexion/reconnexion automatique WebSocket
 * - Gestion des événements de connexion
 * - Parsing et distribution des messages reçus
 * - Gestion des erreurs de connexion
 */

class WebSocketManager {
    constructor(config = {}) {
        const base = config.url || 'ws://localhost:8050/ws/v1/pipeline'; // ✅ endpoint correct
        this.config = {
            url: base,
            reconnectInterval: config.reconnectInterval || 5000,
            maxReconnectAttempts: config.maxReconnectAttempts || 10,
            ...config
        };

        this.ws = null;
        this.reconnectTimer = null;
        this.reconnectAttempts = 0;
        this.isConnected = false;
        this.listeners = new Map();

        // ✅ Bind methods (bien à l’intérieur du constructeur)
        this.connect = this.connect.bind(this);
        this.disconnect = this.disconnect.bind(this);
        this.handleMessage = this.handleMessage.bind(this);
    }

/**
 * Établit la connexion WebSocket
 */
connect() {
        try {
            console.log(`🔌 Tentative de connexion WebSocket: ${this.config.url}`);
            this.ws = new WebSocket(this.config.url);
            
            this.ws.onopen = this.handleOpen.bind(this);
            this.ws.onmessage = this.handleMessage.bind(this);
            this.ws.onclose = this.handleClose.bind(this);
            this.ws.onerror = this.handleError.bind(this);
            
        } catch (error) {
            console.error('❌ Erreur lors de la création WebSocket:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Gère l'ouverture de connexion
     */
    handleOpen() {
        console.log('✅ WebSocket connecté');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.clearReconnectTimer();
        this.emit('connection_status', { status: 'connected' });
    }

    /**
     * Gère la réception des messages
     */
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('📨 Message WebSocket reçu:', data.type || 'unknown');
            console.log('🔍 [DEBUG] Données complètes reçues:', data);
            
            // Émission de l'événement selon le type de message
            if (data.type) {
                console.log(`🚀 [DEBUG] Émission événement '${data.type}' avec données:`, data.data || data);
                this.emit(data.type, data.data || data);
            } else {
                console.log('🚀 [DEBUG] Émission événement "message" avec données:', data);
                this.emit('message', data);
            }
            
        } catch (error) {
            console.error('❌ Erreur parsing message WebSocket:', error);
        }
    }

    /**
     * Gère la fermeture de connexion
     */
    handleClose() {
        console.log('⚠️ WebSocket fermé');
        this.isConnected = false;
        this.emit('connection_status', { status: 'disconnected' });
        this.scheduleReconnect();
    }

    /**
     * Gère les erreurs WebSocket
     */
    handleError(error) {
        console.error('❌ Erreur WebSocket:', error);
        this.emit('connection_status', { status: 'error' });
    }

    /**
     * Planifie une reconnexion automatique
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
            console.error('❌ Nombre maximum de tentatives de reconnexion atteint');
            this.emit('connection_status', { status: 'failed' });
            return;
        }

        this.reconnectAttempts++;
        console.log(`🔄 Reconnexion ${this.reconnectAttempts}/${this.config.maxReconnectAttempts} dans ${this.config.reconnectInterval}ms`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, this.config.reconnectInterval);
    }

    /**
     * Annule le timer de reconnexion
     */
    clearReconnectTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    /**
     * Ferme la connexion WebSocket
     */
    disconnect() {
        console.log('🔌 Fermeture WebSocket demandée');
        this.clearReconnectTimer();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }

    /**
     * Envoie un message via WebSocket
     */
    send(data) {
        if (this.isConnected && this.ws) {
            try {
                this.ws.send(JSON.stringify(data));
                return true;
            } catch (error) {
                console.error('❌ Erreur envoi WebSocket:', error);
                return false;
            }
        }
        console.warn('⚠️ WebSocket non connecté, impossible d\'envoyer');
        return false;
    }

    /**
     * Ajoute un listener pour un type d'événement
     */
    on(eventType, callback) {
        if (!this.listeners.has(eventType)) {
            this.listeners.set(eventType, []);
        }
        this.listeners.get(eventType).push(callback);
    }

    /**
     * Supprime un listener
     */
    off(eventType, callback) {
        const listeners = this.listeners.get(eventType);
        if (listeners) {
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }

    /**
     * Émet un événement vers tous les listeners
     */
    emit(eventType, data) {
        const listeners = this.listeners.get(eventType);
        if (listeners) {
            listeners.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`❌ Erreur dans listener ${eventType}:`, error);
                }
            });
        }
    }

    /**
     * Retourne l'état de la connexion
     */
    getConnectionState() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            maxReconnectAttempts: this.config.maxReconnectAttempts
        };
    }
}

// Export pour utilisation globale
window.WebSocketManager = WebSocketManager;