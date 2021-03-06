#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"

int operation;
std::vector<board::cell> bag;

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }), popup(0, 9) {}

	virtual action take_action(const board& after) {
		//printf("op=%d\n", operation);
		/**********
		printf("environment\n");
		for (int r = 0; r < 4; r++) {
			for (int c = 0; c < 4; c++) {
				printf("%d ", after[r][c]);
			}
			printf("\n");
		}
		**********/
		if (bag.empty()) {
			for (int i = 1; i <= 3; i++)
				bag.push_back(i);
			std::random_shuffle(bag.begin(), bag.end());
		}
		board::cell tile = bag.back();
		bag.pop_back();

		//printf("tile=%d\n", tile);
		//printf("1 %d %d \n", space[0], space[15]);
		std::vector<int> legalspace;
		switch (operation) {
			//case -1: legalspace = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}; break;
			case 0: legalspace = {12, 13, 14, 15}; break;
			case 1: legalspace = {0, 4, 8, 12}; break;
			case 2: legalspace = {0, 1, 2, 3}; break;
			case 3: legalspace = {3, 7, 11, 15}; break;
			default: legalspace = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

		}

		std::shuffle(legalspace.begin(), legalspace.end(), engine);
		//for (int pos : legalspace)
		//	printf("%d ", pos);
		//printf("\n");
		for (int pos : legalspace) {
			if (after(pos) != 0) continue;
			//board::cell tile = popup(engine) ? 1 : 2;
			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;
	std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		/**********
		printf("player\n");
		for (int r = 0; r < 4; r++) {
			for (int c = 0; c < 4; c++) {
				printf("%d ", before[r][c]);
			}
			printf("\n");
		}
		**********/
		board::reward bestreward = -1;
		int bestop = 0;
		for (int op = 0; op < 4; op++) {
			board::reward reward = board(before).slide(op);
			if (reward > bestreward) {
				bestreward = reward;
				bestop = op;
			}
		}
		operation = bestop;
		if (bestreward != -1) return action::slide(bestop);
		return action();
		/*
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			operation = op;	////
			if (reward != -1) return action::slide(op);
		}
		return action();*/
	}

private:
	std::array<int, 4> opcode;
};
