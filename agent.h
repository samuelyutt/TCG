#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

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
 * base agent for agents with weight tables
 */
class weight_agent : public agent {
public:
    weight_agent(const std::string& args = "") : agent(args) {
        if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
            init_weights(meta["init"]);
        if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
            load_weights(meta["load"]);
    }
    virtual ~weight_agent() {
        if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
            save_weights(meta["save"]);
    }

protected:
    virtual void init_weights(const std::string& info) {
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        net.emplace_back(65536); // create an empty weight table with size 65536
        // now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536
    }
    virtual void load_weights(const std::string& path) {
        std::ifstream in(path, std::ios::in | std::ios::binary);
        if (!in.is_open()) std::exit(-1);
        uint32_t size;
        in.read(reinterpret_cast<char*>(&size), sizeof(size));
        net.resize(size);
        for (weight& w : net) in >> w;
        in.close();
    }
    virtual void save_weights(const std::string& path) {
        std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out.is_open()) std::exit(-1);
        uint32_t size = net.size();
        out.write(reinterpret_cast<char*>(&size), sizeof(size));
        for (weight& w : net) out << w;
        out.close();
    }

protected:
    std::vector<weight> net;
};

/**
 * base agent for agents with a learning rate
 */
class learning_agent : public agent {
public:
    learning_agent(const std::string& args = "") : agent(args), alpha(0.1f) {
        if (meta.find("alpha") != meta.end())
            alpha = float(meta["alpha"]);
    }
    virtual ~learning_agent() {}

protected:
    float alpha;
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
        if (bag.empty()) {
            for (int i = 1; i <= 3; i++)
                bag.push_back(i);
            std::random_shuffle(bag.begin(), bag.end());
        }
        board::cell tile = bag.back();
        bag.pop_back();

        std::vector<int> legalspace;
        switch (operation) {
            case 0: legalspace = {12, 13, 14, 15}; break;
            case 1: legalspace = {0, 4, 8, 12}; break;
            case 2: legalspace = {0, 1, 2, 3}; break;
            case 3: legalspace = {3, 7, 11, 15}; break;
            default: legalspace = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        }

        std::shuffle(legalspace.begin(), legalspace.end(), engine);
        
        for (int pos : legalspace) {
            if (after(pos) != 0) continue;
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
class player : public weight_agent {
public:
    player(const std::string& args = "") : weight_agent("name=dummy role=player " + args),
        opcode({ 0, 1, 2, 3 }) {
            for (int i=0; i<8; i++) {
                net.emplace_back(weight(15*15*15*15));
            }
        }

    unsigned encode(const board& state, int t0, int t1, int t2, int t3) const {
        return (state(t0) << 0) | (state(t1) << 4) | (state(t2) << 8) | (state(t3) << 12);
    }

    float get_board_value(const board& state) const {
        float v = 0;
        v += net[0][encode(state, 0, 1, 2, 3)];
        v += net[1][encode(state, 4, 5, 6, 7)];
        v += net[2][encode(state, 8, 9, 10, 11)];
        v += net[3][encode(state, 12, 13, 14, 15)];
        v += net[4][encode(state, 0, 4, 8, 12)];
        v += net[5][encode(state, 1, 5, 9, 13)];
        v += net[6][encode(state, 2, 6, 10, 14)];
        v += net[7][encode(state, 3, 7, 11, 15)];
        //printf("f = %f\n", v);
        return v;
    }

    void train_weight(board::reward reward) {
        double alpha = 0.1/8;
        double v_s = alpha * (get_board_value(next) - get_board_value(previous) + (float)reward/1000);
        if (reward == -1) v_s = 0;
        net[0][encode(previous, 0, 1, 2, 3)] += v_s;
        net[1][encode(previous, 4, 5, 6, 7)] += v_s;
        net[2][encode(previous, 8, 9, 10, 11)] += v_s;
        net[3][encode(previous, 12, 13, 14, 15)] += v_s;
        net[4][encode(previous, 0, 4, 8, 12)] += v_s;
        net[5][encode(previous, 1, 5, 9, 13)] += v_s;
        net[6][encode(previous, 2, 6, 10, 14)] += v_s;
        net[7][encode(previous, 3, 7, 11, 15)] += v_s;
        // printf("next\n");
        // for (int r = 0; r < 4; r++) {
        //         for (int c = 0; c < 4; c++) {
        //             printf("%d ", next[r][c]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // printf("previous\n");
        // for (int r = 0; r < 4; r++) {
        //         for (int c = 0; c < 4; c++) {
        //             printf("%d ", previous[r][c]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        //printf("%f %f %d %f\n", get_board_value(next), get_board_value(previous), reward, v_s);
    }

    virtual void open_episode(const std::string& flag = "") {
        count = 0;
    }

    virtual action take_action(const board& before) {
        //board::reward bestreward = -1;
        float bestvalue = -999999999;
        int bestop = -1;
        int trv, tr;
        for (int op = 0; op < 4; op++) {
            board temp = before;
            board::reward reward = temp.slide(op);
            float value = get_board_value(temp);
            //float value = 0;
            trv = reward + value;
            tr = reward;
            if (bestop == -1 && reward != -1)
                bestop = op;
            if (reward + value > bestvalue && reward != -1) {
                bestvalue = reward + value;
                bestop = op;
            }
        }
        if (bestop != -1) {
            next = before;
            board::reward reward = next.slide(bestop);
            if (count) train_weight(reward);
            previous = next;
            count++;
            return action::slide(operation = bestop);
        } else {
            train_weight(-1);


            //printf("environment\n");
            // printf("%d %d %d\n", bestop, trv, tr);
            // for (int r = 0; r < 4; r++) {
            //     for (int c = 0; c < 4; c++) {
            //         printf("%d ", before[r][c]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");

            return action();
        }
        //if (reward == -1) printf("---------------------------------------------\n");;
        //if (reward != -1) return action::slide(operation = bestop);
        //return action();
    }

private:
    std::array<int, 4> opcode;
    board previous;
    board next;
    int count;
};
