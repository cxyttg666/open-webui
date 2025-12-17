import { WEBUI_API_BASE_URL } from '$lib/constants';

// ==================== Global Connections (Admin) ====================

export interface GlobalConnection {
	id: string;
	name: string;
	type: string; // openai, anthropic, ollama, custom
	url: string;
	api_key?: string; // 管理员配置的API密钥（仅管理员可见）
	auth_type: string;
	config?: any;
	requires_api_key?: boolean; // 是否需要用户提供API密钥
}

export interface GlobalConnectionForm {
	name: string;
	type: string;
	url: string;
	api_key?: string;
	auth_type?: string;
	config?: any;
}

export const getGlobalConnections = async (token: string): Promise<GlobalConnection[]> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/global_connections/`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const createGlobalConnection = async (
	token: string,
	connection: GlobalConnectionForm
): Promise<GlobalConnection> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/global_connections/`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify(connection)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getGlobalConnection = async (
	token: string,
	connectionId: string
): Promise<GlobalConnection> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/global_connections/${connectionId}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const updateGlobalConnection = async (
	token: string,
	connectionId: string,
	connection: Partial<GlobalConnectionForm>
): Promise<GlobalConnection> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/global_connections/${connectionId}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		},
		body: JSON.stringify(connection)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const deleteGlobalConnection = async (
	token: string,
	connectionId: string
): Promise<boolean> => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/global_connections/${connectionId}`, {
		method: 'DELETE',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail;
			return null;
		});

	if (error) {
		throw error;
	}

	return res?.success || false;
};
